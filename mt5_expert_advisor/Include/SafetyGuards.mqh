//+------------------------------------------------------------------+
//|                                              SafetyGuards.mqh    |
//|              Safety guards and fallback mechanisms                |
//|                                     Phase 3 MT5 Integration      |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Neural Bot"
#property version   "2.00"

//+------------------------------------------------------------------+
//| Safety Configuration                                              |
//+------------------------------------------------------------------+
struct SafetyConfig
{
   // Feature validation
   double  max_feature_value;        // Maximum allowed feature value
   double  min_feature_value;        // Minimum allowed feature value
   int     max_nan_features;         // Max NaN features before rejection
   
   // Inference validation
   double  min_confidence;           // Minimum confidence threshold
   double  max_inference_time_ms;    // Max inference time before timeout warning
   int     max_consecutive_errors;   // Max errors before circuit breaker
   
   // Trading guards
   double  max_spread_points;        // Max spread before rejecting trades
   double  min_volume;               // Minimum volume requirement
   bool    require_sequence_full;    // Require full sequence buffer
   
   // Defaults
   void SetDefaults()
   {
      max_feature_value = 1e6;
      min_feature_value = -1e6;
      max_nan_features = 0;
      
      min_confidence = 0.5;
      max_inference_time_ms = 1000;
      max_consecutive_errors = 5;
      
      max_spread_points = 50;
      min_volume = 1000;
      require_sequence_full = true;
   }
};

//+------------------------------------------------------------------+
//| Safety Guard Class                                                |
//+------------------------------------------------------------------+
class CSafetyGuard
{
private:
   SafetyConfig m_config;
   int          m_consecutive_errors;
   bool         m_circuit_breaker_active;
   datetime     m_last_error_time;
   string       m_last_error_message;
   
public:
   //+------------------------------------------------------------------+
   //| Constructor                                                       |
   //+------------------------------------------------------------------+
   CSafetyGuard()
   {
      m_config.SetDefaults();
      m_consecutive_errors = 0;
      m_circuit_breaker_active = false;
      m_last_error_time = 0;
      m_last_error_message = "";
   }
   
   //+------------------------------------------------------------------+
   //| Configure safety parameters                                       |
   //+------------------------------------------------------------------+
   void Configure(const SafetyConfig &config)
   {
      m_config = config;
   }
   
   //+------------------------------------------------------------------+
   //| Validate feature array for NaN/Inf and range                      |
   //+------------------------------------------------------------------+
   bool ValidateFeatures(const float &features[], string &error_msg)
   {
      int nan_count = 0;
      int out_of_range = 0;
      
      int size = ArraySize(features);
      for(int i = 0; i < size; i++)
      {
         if(!MathIsValidNumber(features[i]))
         {
            nan_count++;
            if(nan_count <= 3)
               error_msg += StringFormat("Feature[%d] is NaN/Inf; ", i);
         }
         else if(features[i] > m_config.max_feature_value || 
                 features[i] < m_config.min_feature_value)
         {
            out_of_range++;
            if(out_of_range <= 3)
               error_msg += StringFormat("Feature[%d]=%.4f out of range; ", i, features[i]);
         }
      }
      
      if(nan_count > m_config.max_nan_features)
      {
         error_msg = StringFormat("Too many NaN features: %d (max=%d). ", 
                                  nan_count, m_config.max_nan_features) + error_msg;
         RecordError("Features: " + error_msg);
         return false;
      }
      
      if(out_of_range > size / 10)  // More than 10% out of range
      {
         error_msg = StringFormat("Too many out-of-range features: %d. ", out_of_range) + error_msg;
         RecordError("Features: " + error_msg);
         return false;
      }
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Validate probability outputs                                      |
   //+------------------------------------------------------------------+
   bool ValidateProbabilities(const float &probs[], int expected_classes, string &error_msg)
   {
      int size = ArraySize(probs);
      if(size != expected_classes)
      {
         error_msg = StringFormat("Invalid prob array size: %d (expected %d)", size, expected_classes);
         RecordError("Probs: " + error_msg);
         return false;
      }
      
      double sum = 0;
      for(int i = 0; i < size; i++)
      {
         if(!MathIsValidNumber(probs[i]))
         {
            error_msg = StringFormat("Prob[%d] is NaN/Inf", i);
            RecordError("Probs: " + error_msg);
            return false;
         }
         
         if(probs[i] < 0 || probs[i] > 1)
         {
            error_msg = StringFormat("Prob[%d]=%.4f out of [0,1]", i, probs[i]);
            RecordError("Probs: " + error_msg);
            return false;
         }
         
         sum += probs[i];
      }
      
      if(sum < 0.99 || sum > 1.01)
      {
         error_msg = StringFormat("Prob sum=%.4f (expected ~1.0)", sum);
         RecordError("Probs: " + error_msg);
         return false;
      }
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Validate Transformer output                                       |
   //+------------------------------------------------------------------+
   bool ValidateTransformerOutput(float multi_tf_signal, string &error_msg)
   {
      if(!MathIsValidNumber(multi_tf_signal))
      {
         error_msg = "multi_tf_signal is NaN/Inf";
         RecordError("Transformer: " + error_msg);
         return false;
      }
      
      // Typical range is [-1, 1] but allow some margin
      if(multi_tf_signal < -5.0 || multi_tf_signal > 5.0)
      {
         error_msg = StringFormat("multi_tf_signal=%.4f outside expected range [-5,5]", multi_tf_signal);
         RecordError("Transformer: " + error_msg);
         return false;
      }
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Check trading conditions                                          |
   //+------------------------------------------------------------------+
   bool CanTrade(const string symbol, string &error_msg)
   {
      // Check circuit breaker
      if(m_circuit_breaker_active)
      {
         error_msg = "Circuit breaker active - too many errors";
         return false;
      }
      
      // Check spread
      double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      double spread = (ask - bid) / point;
      
      if(spread > m_config.max_spread_points)
      {
         error_msg = StringFormat("Spread too wide: %.1f points (max=%.1f)", 
                                  spread, m_config.max_spread_points);
         return false;
      }
      
      // Check volume
      long volume = SymbolInfoInteger(symbol, SYMBOL_VOLUME);
      if(volume < m_config.min_volume)
      {
         error_msg = StringFormat("Volume too low: %d (min=%d)", volume, m_config.min_volume);
         return false;
      }
      
      // Check market is open
      if(!SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_FULL)
      {
         error_msg = "Market not fully tradeable";
         return false;
      }
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Record an error (for circuit breaker logic)                       |
   //+------------------------------------------------------------------+
   void RecordError(const string message)
   {
      m_consecutive_errors++;
      m_last_error_time = TimeCurrent();
      m_last_error_message = message;
      
      if(m_consecutive_errors >= m_config.max_consecutive_errors)
      {
         m_circuit_breaker_active = true;
         Print("CIRCUIT BREAKER ACTIVATED: ", m_consecutive_errors, " consecutive errors");
         Print("Last error: ", message);
      }
   }
   
   //+------------------------------------------------------------------+
   //| Record success (resets error counter)                             |
   //+------------------------------------------------------------------+
   void RecordSuccess()
   {
      m_consecutive_errors = 0;
      
      // Auto-reset circuit breaker after 5 minutes of no activity
      if(m_circuit_breaker_active && TimeCurrent() - m_last_error_time > 300)
      {
         m_circuit_breaker_active = false;
         Print("Circuit breaker reset after 5 min timeout");
      }
   }
   
   //+------------------------------------------------------------------+
   //| Reset circuit breaker manually                                    |
   //+------------------------------------------------------------------+
   void ResetCircuitBreaker()
   {
      m_circuit_breaker_active = false;
      m_consecutive_errors = 0;
      Print("Circuit breaker manually reset");
   }
   
   // Accessors
   bool   IsCircuitBreakerActive() const { return m_circuit_breaker_active; }
   int    GetConsecutiveErrors() const { return m_consecutive_errors; }
   string GetLastError() const { return m_last_error_message; }
   
   //+------------------------------------------------------------------+
   //| Get status summary                                                |
   //+------------------------------------------------------------------+
   string GetStatus()
   {
      return StringFormat("SafetyGuard: Errors=%d, CircuitBreaker=%s, LastErr=%s",
                          m_consecutive_errors,
                          m_circuit_breaker_active ? "ACTIVE" : "OFF",
                          m_last_error_message);
   }
};

//+------------------------------------------------------------------+
//| Fallback Handler - provides safe defaults when models fail        |
//+------------------------------------------------------------------+
class CFallbackHandler
{
private:
   bool   m_use_fallback;
   int    m_fallback_signal;     // Default signal when model fails
   double m_fallback_confidence;  // Default confidence
   int    m_fallback_count;       // Number of fallbacks used
   
public:
   CFallbackHandler() : m_use_fallback(true), m_fallback_signal(0), 
                        m_fallback_confidence(0), m_fallback_count(0) {}
   
   //+------------------------------------------------------------------+
   //| Get fallback signal (always HOLD for safety)                      |
   //+------------------------------------------------------------------+
   int GetFallbackSignal(double &confidence, const string reason = "")
   {
      m_fallback_count++;
      confidence = m_fallback_confidence;
      
      if(reason != "")
         Print("FALLBACK activated: ", reason, " (count=", m_fallback_count, ")");
      
      return m_fallback_signal;  // 0 = HOLD
   }
   
   //+------------------------------------------------------------------+
   //| Replace NaN/Inf in array with fallback values                     |
   //+------------------------------------------------------------------+
   void SanitizeFeatures(float &features[], float fallback_value = 0.0f)
   {
      int size = ArraySize(features);
      int replaced = 0;
      
      for(int i = 0; i < size; i++)
      {
         if(!MathIsValidNumber(features[i]))
         {
            features[i] = fallback_value;
            replaced++;
         }
         // Clamp extreme values
         else if(features[i] > 1e6)
         {
            features[i] = 1e6f;
            replaced++;
         }
         else if(features[i] < -1e6)
         {
            features[i] = -1e6f;
            replaced++;
         }
      }
      
      if(replaced > 0)
         PrintFormat("Sanitized %d features with fallback values", replaced);
   }
   
   //+------------------------------------------------------------------+
   //| Replace NaN/Inf probabilities with uniform distribution           |
   //+------------------------------------------------------------------+
   void SanitizeProbabilities(float &probs[], int n_classes = 3)
   {
      ArrayResize(probs, n_classes);
      
      bool needs_fix = false;
      double sum = 0;
      
      for(int i = 0; i < n_classes; i++)
      {
         if(!MathIsValidNumber(probs[i]) || probs[i] < 0 || probs[i] > 1)
         {
            needs_fix = true;
            break;
         }
         sum += probs[i];
      }
      
      if(needs_fix || sum < 0.99 || sum > 1.01)
      {
         // Use uniform distribution as fallback
         float uniform = 1.0f / n_classes;
         for(int i = 0; i < n_classes; i++)
            probs[i] = uniform;
         
         Print("Probabilities sanitized to uniform distribution");
      }
   }
   
   // Accessors
   int GetFallbackCount() const { return m_fallback_count; }
   void ResetFallbackCount() { m_fallback_count = 0; }
   void EnableFallback(bool enable) { m_use_fallback = enable; }
};

//+------------------------------------------------------------------+
//| Performance Monitor                                               |
//+------------------------------------------------------------------+
class CPerformanceMonitor
{
private:
   ulong  m_start_time;
   double m_total_inference_time;
   int    m_inference_count;
   double m_max_inference_time;
   
public:
   CPerformanceMonitor() : m_start_time(0), m_total_inference_time(0), 
                           m_inference_count(0), m_max_inference_time(0) {}
   
   void StartTimer()
   {
      m_start_time = GetMicrosecondCount();
   }
   
   double StopTimer()
   {
      if(m_start_time == 0) return 0;
      
      ulong elapsed = GetMicrosecondCount() - m_start_time;
      double elapsed_ms = elapsed / 1000.0;
      
      m_total_inference_time += elapsed_ms;
      m_inference_count++;
      
      if(elapsed_ms > m_max_inference_time)
         m_max_inference_time = elapsed_ms;
      
      m_start_time = 0;
      return elapsed_ms;
   }
   
   double GetAverageTime() const
   {
      return m_inference_count > 0 ? m_total_inference_time / m_inference_count : 0;
   }
   
   double GetMaxTime() const { return m_max_inference_time; }
   int GetCount() const { return m_inference_count; }
   
   string GetStats()
   {
      return StringFormat("Inference: count=%d, avg=%.2fms, max=%.2fms",
                          m_inference_count, GetAverageTime(), m_max_inference_time);
   }
};
