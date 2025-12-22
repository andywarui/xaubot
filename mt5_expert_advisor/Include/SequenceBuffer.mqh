//+------------------------------------------------------------------+
//|                                              SequenceBuffer.mqh  |
//|              Rolling buffer for Transformer sequence input       |
//|                                     Phase 3 MT5 Integration      |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Neural Bot"
#property version   "2.00"

//+------------------------------------------------------------------+
//| Sequence Buffer Class                                             |
//| Maintains a rolling window of scaled feature vectors for         |
//| Transformer input [seq_len, n_features]                           |
//+------------------------------------------------------------------+
class CSequenceBuffer
{
private:
   float  m_buffer[];        // Flattened buffer [seq_len * n_features]
   int    m_seq_len;         // Sequence length (e.g., 30)
   int    m_n_features;      // Features per timestep (e.g., 130)
   int    m_current_count;   // Number of timesteps filled
   bool   m_is_full;         // Buffer has seq_len timesteps
   
public:
   //+------------------------------------------------------------------+
   //| Constructor                                                       |
   //+------------------------------------------------------------------+
   CSequenceBuffer() : m_seq_len(30), m_n_features(130), m_current_count(0), m_is_full(false)
   {
      ArrayResize(m_buffer, m_seq_len * m_n_features);
      ArrayInitialize(m_buffer, 0.0f);
   }
   
   //+------------------------------------------------------------------+
   //| Initialize with specific dimensions                               |
   //+------------------------------------------------------------------+
   bool Init(const int seq_len, const int n_features)
   {
      if(seq_len <= 0 || n_features <= 0) return false;
      
      m_seq_len = seq_len;
      m_n_features = n_features;
      m_current_count = 0;
      m_is_full = false;
      
      ArrayResize(m_buffer, m_seq_len * m_n_features);
      ArrayInitialize(m_buffer, 0.0f);
      
      PrintFormat("SequenceBuffer initialized: seq_len=%d, features=%d, total=%d", 
                  m_seq_len, m_n_features, m_seq_len * m_n_features);
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Reset buffer to initial state                                     |
   //+------------------------------------------------------------------+
   void Reset()
   {
      m_current_count = 0;
      m_is_full = false;
      ArrayInitialize(m_buffer, 0.0f);
   }
   
   //+------------------------------------------------------------------+
   //| Push new feature vector (shifts buffer, adds to end)              |
   //| Features should already be scaled                                 |
   //+------------------------------------------------------------------+
   bool Push(const float &features[])
   {
      int size = ArraySize(features);
      if(size != m_n_features)
      {
         PrintFormat("ERROR: Push size mismatch - expected %d, got %d", m_n_features, size);
         return false;
      }
      
      // Shift buffer: move all rows up by 1 (drop oldest at position 0)
      // Buffer layout: [t0_f0, t0_f1, ..., t0_fn, t1_f0, ..., tN_fn]
      // We shift everything left by n_features to make room at the end
      
      for(int i = 0; i < (m_seq_len - 1) * m_n_features; i++)
      {
         m_buffer[i] = m_buffer[i + m_n_features];
      }
      
      // Add new features at the end (most recent timestep)
      int start_idx = (m_seq_len - 1) * m_n_features;
      for(int i = 0; i < m_n_features; i++)
      {
         m_buffer[start_idx + i] = features[i];
      }
      
      // Update count
      if(m_current_count < m_seq_len)
         m_current_count++;
      
      if(m_current_count >= m_seq_len)
         m_is_full = true;
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Get buffer as 1D array for ONNX input [1, seq_len, n_features]    |
   //| Returns copy reshaped for ONNX (row-major order)                  |
   //+------------------------------------------------------------------+
   bool GetOnnxInput(float &output[])
   {
      if(!m_is_full)
      {
         PrintFormat("WARNING: Buffer not full (%d/%d timesteps)", m_current_count, m_seq_len);
         // Can still return partial buffer for warm-up
      }
      
      int total = m_seq_len * m_n_features;
      ArrayResize(output, total);
      ArrayCopy(output, m_buffer, 0, 0, total);
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Get feature at specific timestep and feature index                |
   //| timestep: 0 = oldest, seq_len-1 = most recent                    |
   //+------------------------------------------------------------------+
   float GetValue(const int timestep, const int feature_idx)
   {
      if(timestep < 0 || timestep >= m_seq_len) return 0.0f;
      if(feature_idx < 0 || feature_idx >= m_n_features) return 0.0f;
      
      return m_buffer[timestep * m_n_features + feature_idx];
   }
   
   //+------------------------------------------------------------------+
   //| Set feature at specific timestep and feature index                |
   //+------------------------------------------------------------------+
   void SetValue(const int timestep, const int feature_idx, const float value)
   {
      if(timestep < 0 || timestep >= m_seq_len) return;
      if(feature_idx < 0 || feature_idx >= m_n_features) return;
      
      m_buffer[timestep * m_n_features + feature_idx] = value;
   }
   
   //+------------------------------------------------------------------+
   //| Fill buffer from historical data (warm-up)                        |
   //| Requires callback to calculate features for each shift            |
   //+------------------------------------------------------------------+
   bool WarmUp(CFeatureCalculator &calc, CMinMaxScaler &scaler, const int start_shift)
   {
      Reset();
      
      // Fill from oldest to newest
      // start_shift should be large enough to fill seq_len bars
      // e.g., start_shift = 30 means we load shifts 30, 29, 28, ... 1
      
      for(int i = m_seq_len - 1; i >= 0; i--)
      {
         int shift = start_shift - i;
         if(shift < 1) shift = 1;  // Minimum shift is 1 (last closed bar)
         
         float raw_features[];
         if(!calc.CalculateTransformerFeatures(raw_features, shift))
         {
            PrintFormat("WARNING: Failed to calculate features at shift %d", shift);
            // Fill with zeros
            ArrayResize(raw_features, m_n_features);
            ArrayInitialize(raw_features, 0.0f);
         }
         
         // Scale features
         float scaled_features[];
         if(scaler.IsLoaded())
         {
            if(!scaler.Transform(raw_features, scaled_features))
            {
               PrintFormat("WARNING: Failed to scale features at shift %d", shift);
               ArrayCopy(scaled_features, raw_features);
            }
         }
         else
         {
            ArrayCopy(scaled_features, raw_features);
         }
         
         // Push to buffer
         Push(scaled_features);
      }
      
      PrintFormat("SequenceBuffer warm-up complete: %d/%d timesteps", m_current_count, m_seq_len);
      return m_is_full;
   }
   
   // Accessors
   bool IsFull() const { return m_is_full; }
   int  GetSeqLen() const { return m_seq_len; }
   int  GetNFeatures() const { return m_n_features; }
   int  GetCurrentCount() const { return m_current_count; }
   int  GetTotalSize() const { return m_seq_len * m_n_features; }
   
   //+------------------------------------------------------------------+
   //| Debug: Print buffer statistics                                    |
   //+------------------------------------------------------------------+
   void PrintStats()
   {
      if(ArraySize(m_buffer) == 0)
      {
         Print("SequenceBuffer: Empty");
         return;
      }
      
      float min_val = m_buffer[0];
      float max_val = m_buffer[0];
      float sum = 0;
      
      int total = m_seq_len * m_n_features;
      for(int i = 0; i < total; i++)
      {
         if(m_buffer[i] < min_val) min_val = m_buffer[i];
         if(m_buffer[i] > max_val) max_val = m_buffer[i];
         sum += m_buffer[i];
      }
      
      float mean = sum / total;
      
      PrintFormat("SequenceBuffer: %s, count=%d/%d", m_is_full ? "FULL" : "FILLING", m_current_count, m_seq_len);
      PrintFormat("  Size: [%d x %d] = %d values", m_seq_len, m_n_features, total);
      PrintFormat("  Range: [%.6f, %.6f], Mean: %.6f", min_val, max_val, mean);
      
      // Print first and last timestep sample
      if(m_is_full)
      {
         PrintFormat("  First timestep [0]: [%.4f, %.4f, %.4f, ...]", 
                     m_buffer[0], m_buffer[1], m_buffer[2]);
         int last_start = (m_seq_len - 1) * m_n_features;
         PrintFormat("  Last timestep [%d]: [%.4f, %.4f, %.4f, ...]", 
                     m_seq_len - 1, m_buffer[last_start], m_buffer[last_start+1], m_buffer[last_start+2]);
      }
   }
};

//+------------------------------------------------------------------+
//| Helper: ONNX inference wrapper for Transformer                    |
//+------------------------------------------------------------------+
class CTransformerInference
{
private:
   long   m_handle;
   bool   m_ready;
   int    m_seq_len;
   int    m_n_features;
   string m_input_name;
   string m_output_name;
   
public:
   CTransformerInference() : m_handle(INVALID_HANDLE), m_ready(false), m_seq_len(30), m_n_features(130) {}
   ~CTransformerInference() { Release(); }
   
   //+------------------------------------------------------------------+
   //| Load Transformer ONNX model                                       |
   //+------------------------------------------------------------------+
   bool Load(const string model_file, const int seq_len = 30, const int n_features = 130, const bool debug = false)
   {
      m_seq_len = seq_len;
      m_n_features = n_features;
      
      uint flags = debug ? ONNX_DEBUG_LOGS : ONNX_DEFAULT;
      
      // Try multiple paths
      string candidates[] = {
         model_file,
         "NeuralBot\\" + model_file,
         "Files\\" + model_file,
         "Files\\NeuralBot\\" + model_file
      };
      
      for(int i = 0; i < ArraySize(candidates); i++)
      {
         ResetLastError();
         m_handle = OnnxCreate(candidates[i], flags);
         if(m_handle != INVALID_HANDLE)
         {
            PrintFormat("Transformer loaded from: %s", candidates[i]);
            break;
         }
      }
      
      if(m_handle == INVALID_HANDLE)
      {
         Print("ERROR: Failed to load Transformer ONNX");
         return false;
      }
      
      // Set input shape [1, seq_len, n_features]
      ulong input_shape[] = {1, (ulong)m_seq_len, (ulong)m_n_features};
      if(!OnnxSetInputShape(m_handle, 0, input_shape))
      {
         PrintFormat("ERROR: Failed to set Transformer input shape [1, %d, %d]", m_seq_len, m_n_features);
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }
      
      // Set output shape [1, 1]
      ulong output_shape[] = {1, 1};
      if(!OnnxSetOutputShape(m_handle, 0, output_shape))
      {
         Print("ERROR: Failed to set Transformer output shape [1, 1]");
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }
      
      m_ready = true;
      PrintFormat("Transformer ready: input [1, %d, %d], output [1, 1]", m_seq_len, m_n_features);
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Release model                                                     |
   //+------------------------------------------------------------------+
   void Release()
   {
      if(m_handle != INVALID_HANDLE)
      {
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
      }
      m_ready = false;
   }
   
   //+------------------------------------------------------------------+
   //| Run inference and get multi_tf_signal                             |
   //+------------------------------------------------------------------+
   bool Predict(const float &input[], float &multi_tf_signal)
   {
      multi_tf_signal = 0.0f;
      
      if(!m_ready || m_handle == INVALID_HANDLE)
      {
         Print("ERROR: Transformer not ready");
         return false;
      }
      
      int expected_size = m_seq_len * m_n_features;
      if(ArraySize(input) != expected_size)
      {
         PrintFormat("ERROR: Input size mismatch - expected %d, got %d", expected_size, ArraySize(input));
         return false;
      }
      
      // Prepare output buffer
      static float output_buffer[1];
      
      // Run inference
      if(!OnnxRun(m_handle, ONNX_DEFAULT, input, output_buffer))
      {
         int error = GetLastError();
         PrintFormat("ERROR: Transformer inference failed (code=%d)", error);
         return false;
      }
      
      multi_tf_signal = output_buffer[0];
      return true;
   }
   
   bool IsReady() const { return m_ready; }
   long GetHandle() const { return m_handle; }
};

//+------------------------------------------------------------------+
//| Helper: ONNX inference wrapper for LightGBM                       |
//+------------------------------------------------------------------+
class CLightGBMInference
{
private:
   long   m_handle;
   bool   m_ready;
   int    m_n_features;  // 27 for hybrid (with multi_tf_signal)
   
public:
   CLightGBMInference() : m_handle(INVALID_HANDLE), m_ready(false), m_n_features(27) {}
   ~CLightGBMInference() { Release(); }
   
   //+------------------------------------------------------------------+
   //| Load LightGBM ONNX model                                          |
   //+------------------------------------------------------------------+
   bool Load(const string model_file, const int n_features = 27, const bool debug = false)
   {
      m_n_features = n_features;
      
      uint flags = debug ? ONNX_DEBUG_LOGS : ONNX_DEFAULT;
      
      // Try multiple paths
      string candidates[] = {
         model_file,
         "NeuralBot\\" + model_file,
         "Files\\" + model_file,
         "Files\\NeuralBot\\" + model_file
      };
      
      for(int i = 0; i < ArraySize(candidates); i++)
      {
         ResetLastError();
         m_handle = OnnxCreate(candidates[i], flags);
         if(m_handle != INVALID_HANDLE)
         {
            PrintFormat("LightGBM loaded from: %s", candidates[i]);
            break;
         }
      }
      
      if(m_handle == INVALID_HANDLE)
      {
         Print("ERROR: Failed to load LightGBM ONNX");
         return false;
      }
      
      // Set input shape [1, n_features]
      ulong input_shape[] = {1, (ulong)m_n_features};
      if(!OnnxSetInputShape(m_handle, 0, input_shape))
      {
         PrintFormat("ERROR: Failed to set LightGBM input shape [1, %d]", m_n_features);
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }
      
      // Set output shapes
      // Output 0: label [1]
      ulong label_shape[] = {1};
      if(!OnnxSetOutputShape(m_handle, 0, label_shape))
      {
         Print("ERROR: Failed to set LightGBM output[0] shape");
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }
      
      // Output 1: probabilities [1, 3]
      ulong probs_shape[] = {1, 3};
      if(!OnnxSetOutputShape(m_handle, 1, probs_shape))
      {
         Print("ERROR: Failed to set LightGBM output[1] shape");
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }
      
      m_ready = true;
      PrintFormat("LightGBM ready: input [1, %d], outputs [1], [1, 3]", m_n_features);
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Release model                                                     |
   //+------------------------------------------------------------------+
   void Release()
   {
      if(m_handle != INVALID_HANDLE)
      {
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
      }
      m_ready = false;
   }
   
   //+------------------------------------------------------------------+
   //| Run inference                                                     |
   //| Returns: label (0=HOLD, 1=BUY, 2=SELL) and probabilities          |
   //+------------------------------------------------------------------+
   bool Predict(const float &input[], int &label, float &probs[])
   {
      label = 0;  // Default HOLD
      ArrayResize(probs, 3);
      ArrayInitialize(probs, 0.0f);
      
      if(!m_ready || m_handle == INVALID_HANDLE)
      {
         Print("ERROR: LightGBM not ready");
         return false;
      }
      
      if(ArraySize(input) != m_n_features)
      {
         PrintFormat("ERROR: Input size mismatch - expected %d, got %d", m_n_features, ArraySize(input));
         return false;
      }
      
      // Prepare output buffers
      static long output_label[1];
      static float output_probs[3];
      
      // Run inference
      if(!OnnxRun(m_handle, ONNX_DEFAULT, input, output_label, output_probs))
      {
         int error = GetLastError();
         PrintFormat("ERROR: LightGBM inference failed (code=%d)", error);
         return false;
      }
      
      label = (int)output_label[0];
      ArrayCopy(probs, output_probs, 0, 0, 3);
      
      return true;
   }
   
   bool IsReady() const { return m_ready; }
   long GetHandle() const { return m_handle; }
};
