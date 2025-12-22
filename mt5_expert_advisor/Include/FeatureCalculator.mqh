//+------------------------------------------------------------------+
//|                                           FeatureCalculator.mqh  |
//|            Feature calculation for hybrid Transformer+LightGBM   |
//|                                     Phase 3 MT5 Integration      |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Neural Bot"
#property version   "2.00"

//+------------------------------------------------------------------+
//| Feature names for the 26 LightGBM features (multi_tf_signal=0)   |
//+------------------------------------------------------------------+
string FEATURE_NAMES_26[26] = {
   "body","body_abs","candle_range","close_position",
   "return_1","return_5","return_15","return_60",
   "tr","atr_14","rsi_14","ema_10","ema_20","ema_50",
   "hour_sin","hour_cos","M5_trend","M5_position",
   "M15_trend","M15_position","H1_trend","H1_position",
   "H4_trend","H4_position","D1_trend","D1_position"
};

//+------------------------------------------------------------------+
//| Feature names for 27 hybrid features (multi_tf_signal at index 0)|
//+------------------------------------------------------------------+
string FEATURE_NAMES_27[27] = {
   "multi_tf_signal",  // Index 0: From Transformer
   "body","body_abs","candle_range","close_position",
   "return_1","return_5","return_15","return_60",
   "tr","atr_14","rsi_14","ema_10","ema_20","ema_50",
   "hour_sin","hour_cos","M5_trend","M5_position",
   "M15_trend","M15_position","H1_trend","H1_position",
   "H4_trend","H4_position","D1_trend","D1_position"
};

//+------------------------------------------------------------------+
//| Scaler parameters for Transformer (130 features)                 |
//+------------------------------------------------------------------+
class CMinMaxScaler
{
private:
   double m_scale[];      // scale factors
   double m_min[];        // min offsets
   int    m_n_features;   // number of features
   bool   m_loaded;       // successfully loaded
   
public:
   CMinMaxScaler() : m_n_features(0), m_loaded(false) {}
   ~CMinMaxScaler() {}
   
   //+------------------------------------------------------------------+
   //| Load scaler parameters from JSON file                            |
   //+------------------------------------------------------------------+
   bool Load(const string filename)
   {
      int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
      if(handle == INVALID_HANDLE)
      {
         Print("ERROR: Cannot open scaler file: ", filename);
         return false;
      }
      
      // Read entire file
      string content = "";
      while(!FileIsEnding(handle))
      {
         content += FileReadString(handle) + "\n";
      }
      FileClose(handle);
      
      // Parse n_features
      int pos = StringFind(content, "\"n_features\":");
      if(pos < 0) return false;
      int start = pos + 13;
      int end = StringFind(content, ",", start);
      m_n_features = (int)StringToInteger(StringSubstr(content, start, end - start));
      
      // Allocate arrays
      ArrayResize(m_scale, m_n_features);
      ArrayResize(m_min, m_n_features);
      
      // Parse scale array
      if(!ParseJsonArray(content, "\"scale\":", m_scale, m_n_features))
         return false;
         
      // Parse min array
      if(!ParseJsonArray(content, "\"min\":", m_min, m_n_features))
         return false;
      
      m_loaded = true;
      Print("Scaler loaded: ", m_n_features, " features");
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Apply scaling: X_scaled = X * scale + min                        |
   //+------------------------------------------------------------------+
   bool Transform(const float &input[], float &output[])
   {
      if(!m_loaded) return false;
      
      int size = ArraySize(input);
      if(size != m_n_features)
      {
         PrintFormat("ERROR: Scaler size mismatch - expected %d, got %d", m_n_features, size);
         return false;
      }
      
      ArrayResize(output, m_n_features);
      for(int i = 0; i < m_n_features; i++)
      {
         output[i] = (float)(input[i] * m_scale[i] + m_min[i]);
      }
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Transform in-place                                               |
   //+------------------------------------------------------------------+
   bool TransformInPlace(float &data[])
   {
      if(!m_loaded) return false;
      
      int size = ArraySize(data);
      if(size != m_n_features)
      {
         PrintFormat("ERROR: Scaler size mismatch - expected %d, got %d", m_n_features, size);
         return false;
      }
      
      for(int i = 0; i < m_n_features; i++)
      {
         data[i] = (float)(data[i] * m_scale[i] + m_min[i]);
      }
      
      return true;
   }
   
   bool IsLoaded() const { return m_loaded; }
   int  GetNFeatures() const { return m_n_features; }
   
private:
   //+------------------------------------------------------------------+
   //| Parse JSON array of doubles                                      |
   //+------------------------------------------------------------------+
   bool ParseJsonArray(const string &content, const string key, double &arr[], const int expected_size)
   {
      int pos = StringFind(content, key);
      if(pos < 0) return false;
      
      int arr_start = StringFind(content, "[", pos);
      int arr_end = StringFind(content, "]", arr_start);
      
      if(arr_start < 0 || arr_end < 0) return false;
      
      string arr_str = StringSubstr(content, arr_start + 1, arr_end - arr_start - 1);
      
      // Split by comma and parse
      int idx = 0;
      int search_pos = 0;
      
      while(idx < expected_size)
      {
         int comma_pos = StringFind(arr_str, ",", search_pos);
         string value_str;
         
         if(comma_pos < 0)
            value_str = StringSubstr(arr_str, search_pos);
         else
            value_str = StringSubstr(arr_str, search_pos, comma_pos - search_pos);
         
         // Trim whitespace
         StringTrimLeft(value_str);
         StringTrimRight(value_str);
         
         arr[idx] = StringToDouble(value_str);
         idx++;
         
         if(comma_pos < 0) break;
         search_pos = comma_pos + 1;
      }
      
      return (idx == expected_size);
   }
};

//+------------------------------------------------------------------+
//| Feature Calculator Class                                          |
//| Calculates all 130 features for Transformer input                 |
//+------------------------------------------------------------------+
class CFeatureCalculator
{
private:
   // Indicator handles (M1 timeframe)
   int m_atr14_m1_handle;
   int m_rsi14_m1_handle;
   int m_ema10_m1_handle;
   int m_ema20_m1_handle;
   int m_ema50_m1_handle;
   
   // Multi-timeframe EMA handles
   int m_ema20_m5_handle;
   int m_ema20_m15_handle;
   int m_ema20_h1_handle;
   int m_ema20_h4_handle;
   int m_ema20_d1_handle;
   
   string m_symbol;
   bool   m_initialized;
   
public:
   CFeatureCalculator() : m_initialized(false) {}
   ~CFeatureCalculator() { Deinit(); }
   
   //+------------------------------------------------------------------+
   //| Initialize indicator handles                                     |
   //+------------------------------------------------------------------+
   bool Init(const string symbol)
   {
      m_symbol = symbol;
      
      // M1 indicators
      m_atr14_m1_handle = iATR(m_symbol, PERIOD_M1, 14);
      m_rsi14_m1_handle = iRSI(m_symbol, PERIOD_M1, 14, PRICE_CLOSE);
      m_ema10_m1_handle = iMA(m_symbol, PERIOD_M1, 10, 0, MODE_EMA, PRICE_CLOSE);
      m_ema20_m1_handle = iMA(m_symbol, PERIOD_M1, 20, 0, MODE_EMA, PRICE_CLOSE);
      m_ema50_m1_handle = iMA(m_symbol, PERIOD_M1, 50, 0, MODE_EMA, PRICE_CLOSE);
      
      // Multi-timeframe EMAs
      m_ema20_m5_handle  = iMA(m_symbol, PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE);
      m_ema20_m15_handle = iMA(m_symbol, PERIOD_M15, 20, 0, MODE_EMA, PRICE_CLOSE);
      m_ema20_h1_handle  = iMA(m_symbol, PERIOD_H1, 20, 0, MODE_EMA, PRICE_CLOSE);
      m_ema20_h4_handle  = iMA(m_symbol, PERIOD_H4, 20, 0, MODE_EMA, PRICE_CLOSE);
      m_ema20_d1_handle  = iMA(m_symbol, PERIOD_D1, 20, 0, MODE_EMA, PRICE_CLOSE);
      
      // Validate all handles
      if(m_atr14_m1_handle == INVALID_HANDLE || 
         m_rsi14_m1_handle == INVALID_HANDLE ||
         m_ema10_m1_handle == INVALID_HANDLE || 
         m_ema20_m1_handle == INVALID_HANDLE || 
         m_ema50_m1_handle == INVALID_HANDLE ||
         m_ema20_m5_handle == INVALID_HANDLE || 
         m_ema20_m15_handle == INVALID_HANDLE || 
         m_ema20_h1_handle == INVALID_HANDLE ||
         m_ema20_h4_handle == INVALID_HANDLE || 
         m_ema20_d1_handle == INVALID_HANDLE)
      {
         Print("ERROR: Failed to create indicator handles");
         return false;
      }
      
      m_initialized = true;
      Print("FeatureCalculator initialized for ", m_symbol);
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Release indicator handles                                        |
   //+------------------------------------------------------------------+
   void Deinit()
   {
      if(m_atr14_m1_handle != INVALID_HANDLE) { IndicatorRelease(m_atr14_m1_handle); m_atr14_m1_handle = INVALID_HANDLE; }
      if(m_rsi14_m1_handle != INVALID_HANDLE) { IndicatorRelease(m_rsi14_m1_handle); m_rsi14_m1_handle = INVALID_HANDLE; }
      if(m_ema10_m1_handle != INVALID_HANDLE) { IndicatorRelease(m_ema10_m1_handle); m_ema10_m1_handle = INVALID_HANDLE; }
      if(m_ema20_m1_handle != INVALID_HANDLE) { IndicatorRelease(m_ema20_m1_handle); m_ema20_m1_handle = INVALID_HANDLE; }
      if(m_ema50_m1_handle != INVALID_HANDLE) { IndicatorRelease(m_ema50_m1_handle); m_ema50_m1_handle = INVALID_HANDLE; }
      
      if(m_ema20_m5_handle != INVALID_HANDLE)  { IndicatorRelease(m_ema20_m5_handle);  m_ema20_m5_handle = INVALID_HANDLE; }
      if(m_ema20_m15_handle != INVALID_HANDLE) { IndicatorRelease(m_ema20_m15_handle); m_ema20_m15_handle = INVALID_HANDLE; }
      if(m_ema20_h1_handle != INVALID_HANDLE)  { IndicatorRelease(m_ema20_h1_handle);  m_ema20_h1_handle = INVALID_HANDLE; }
      if(m_ema20_h4_handle != INVALID_HANDLE)  { IndicatorRelease(m_ema20_h4_handle);  m_ema20_h4_handle = INVALID_HANDLE; }
      if(m_ema20_d1_handle != INVALID_HANDLE)  { IndicatorRelease(m_ema20_d1_handle);  m_ema20_d1_handle = INVALID_HANDLE; }
      
      m_initialized = false;
   }
   
   //+------------------------------------------------------------------+
   //| Calculate 26 features for LightGBM (without multi_tf_signal)    |
   //| Returns features in order matching training data                 |
   //+------------------------------------------------------------------+
   bool CalculateLightGBMFeatures(float &features[], const int shift)
   {
      if(!m_initialized) return false;
      
      ArrayResize(features, 26);
      
      // Get M1 OHLC
      double o = iOpen(m_symbol, PERIOD_M1, shift);
      double h = iHigh(m_symbol, PERIOD_M1, shift);
      double l = iLow(m_symbol, PERIOD_M1, shift);
      double c = iClose(m_symbol, PERIOD_M1, shift);
      
      if(o <= 0 || h <= 0 || l <= 0 || c <= 0) return false;
      
      int idx = 0;
      
      // 0-3: Price features
      features[idx++] = (float)(c - o);                                    // body
      features[idx++] = (float)MathAbs(c - o);                             // body_abs
      features[idx++] = (float)(h - l);                                    // candle_range
      features[idx++] = (float)((c - l) / (h - l + 0.0001));              // close_position
      
      // 4-7: Returns
      double c1 = iClose(m_symbol, PERIOD_M1, shift + 1);
      double c5 = iClose(m_symbol, PERIOD_M1, shift + 5);
      double c15 = iClose(m_symbol, PERIOD_M1, shift + 15);
      double c60 = iClose(m_symbol, PERIOD_M1, shift + 60);
      
      features[idx++] = (float)((c - c1) / (c1 + 0.0001));                // return_1
      features[idx++] = (float)((c - c5) / (c5 + 0.0001));                // return_5
      features[idx++] = (float)((c - c15) / (c15 + 0.0001));              // return_15
      features[idx++] = (float)((c - c60) / (c60 + 0.0001));              // return_60
      
      // 8-9: ATR
      double prev_close = iClose(m_symbol, PERIOD_M1, shift + 1);
      double tr = MathMax(h - l, MathMax(MathAbs(h - prev_close), MathAbs(l - prev_close)));
      features[idx++] = (float)tr;                                         // tr
      
      double atr14_buffer[];
      ArraySetAsSeries(atr14_buffer, true);
      if(CopyBuffer(m_atr14_m1_handle, 0, shift, 1, atr14_buffer) <= 0) return false;
      features[idx++] = (float)atr14_buffer[0];                            // atr_14
      
      // 10: RSI
      double rsi14_buffer[];
      ArraySetAsSeries(rsi14_buffer, true);
      if(CopyBuffer(m_rsi14_m1_handle, 0, shift, 1, rsi14_buffer) <= 0) return false;
      features[idx++] = (float)rsi14_buffer[0];                            // rsi_14
      
      // 11-13: EMAs
      double ema10_buffer[], ema20_buffer[], ema50_buffer[];
      ArraySetAsSeries(ema10_buffer, true);
      ArraySetAsSeries(ema20_buffer, true);
      ArraySetAsSeries(ema50_buffer, true);

      if(CopyBuffer(m_ema10_m1_handle, 0, shift, 1, ema10_buffer) <= 0) return false;
      if(CopyBuffer(m_ema20_m1_handle, 0, shift, 1, ema20_buffer) <= 0) return false;
      if(CopyBuffer(m_ema50_m1_handle, 0, shift, 1, ema50_buffer) <= 0) return false;
      
      features[idx++] = (float)ema10_buffer[0];                            // ema_10
      features[idx++] = (float)ema20_buffer[0];                            // ema_20
      features[idx++] = (float)ema50_buffer[0];                            // ema_50
      
      // 14-15: Time features
      MqlDateTime dt;
      TimeToStruct(iTime(m_symbol, PERIOD_M1, shift), dt);
      features[idx++] = (float)MathSin(2 * M_PI * dt.hour / 24);          // hour_sin
      features[idx++] = (float)MathCos(2 * M_PI * dt.hour / 24);          // hour_cos
      
      // 16-25: Multi-timeframe context
      if(!CalculateMTFFeatures(features, idx, shift)) return false;
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Calculate 130 features for Transformer input at given shift     |
   //| These are RAW features that will be scaled by MinMaxScaler      |
   //+------------------------------------------------------------------+
   bool CalculateTransformerFeatures(float &features[], const int shift)
   {
      if(!m_initialized) return false;
      
      // Transformer uses 130 features per timestep
      // This includes extended features compared to LightGBM
      ArrayResize(features, 130);
      
      // Get M1 OHLC for current shift
      double o = iOpen(m_symbol, PERIOD_M1, shift);
      double h = iHigh(m_symbol, PERIOD_M1, shift);
      double l = iLow(m_symbol, PERIOD_M1, shift);
      double c = iClose(m_symbol, PERIOD_M1, shift);
      
      if(o <= 0 || h <= 0 || l <= 0 || c <= 0) return false;
      
      int idx = 0;
      
      // Core price features (same as LightGBM base)
      features[idx++] = (float)(c - o);                                    // body
      features[idx++] = (float)MathAbs(c - o);                             // body_abs
      features[idx++] = (float)(h - l);                                    // candle_range
      features[idx++] = (float)((c - l) / (h - l + 0.0001));              // close_position
      
      // Returns at multiple lookbacks
      for(int lb = 1; lb <= 60; lb++)  // Extended returns: 1-60 periods
      {
         double c_lb = iClose(m_symbol, PERIOD_M1, shift + lb);
         if(c_lb > 0)
            features[idx++] = (float)((c - c_lb) / (c_lb + 0.0001));
         else
            features[idx++] = 0.0f;
      }
      
      // ATR and TR
      double prev_close = iClose(m_symbol, PERIOD_M1, shift + 1);
      double tr = MathMax(h - l, MathMax(MathAbs(h - prev_close), MathAbs(l - prev_close)));
      features[idx++] = (float)tr;
      
      double atr14_buffer[];
      ArraySetAsSeries(atr14_buffer, true);
      if(CopyBuffer(m_atr14_m1_handle, 0, shift, 1, atr14_buffer) <= 0) 
         features[idx++] = 0.0f;
      else
         features[idx++] = (float)atr14_buffer[0];
      
      // RSI
      double rsi14_buffer[];
      ArraySetAsSeries(rsi14_buffer, true);
      if(CopyBuffer(m_rsi14_m1_handle, 0, shift, 1, rsi14_buffer) <= 0)
         features[idx++] = 50.0f;  // Neutral RSI
      else
         features[idx++] = (float)rsi14_buffer[0];
      
      // EMAs
      double ema_buffers[3][];
      for(int i = 0; i < 3; i++) ArraySetAsSeries(ema_buffers[i], true);
      
      if(CopyBuffer(m_ema10_m1_handle, 0, shift, 1, ema_buffers[0]) <= 0) ema_buffers[0][0] = c;
      if(CopyBuffer(m_ema20_m1_handle, 0, shift, 1, ema_buffers[1]) <= 0) ema_buffers[1][0] = c;
      if(CopyBuffer(m_ema50_m1_handle, 0, shift, 1, ema_buffers[2]) <= 0) ema_buffers[2][0] = c;
      
      features[idx++] = (float)ema_buffers[0][0];
      features[idx++] = (float)ema_buffers[1][0];
      features[idx++] = (float)ema_buffers[2][0];
      
      // Time features
      MqlDateTime dt;
      TimeToStruct(iTime(m_symbol, PERIOD_M1, shift), dt);
      features[idx++] = (float)MathSin(2 * M_PI * dt.hour / 24);
      features[idx++] = (float)MathCos(2 * M_PI * dt.hour / 24);
      features[idx++] = (float)MathSin(2 * M_PI * dt.day_of_week / 7);
      features[idx++] = (float)MathCos(2 * M_PI * dt.day_of_week / 7);
      features[idx++] = (float)MathSin(2 * M_PI * dt.min / 60);
      features[idx++] = (float)MathCos(2 * M_PI * dt.min / 60);
      
      // Multi-timeframe features (extended)
      if(!CalculateExtendedMTFFeatures(features, idx, shift)) 
      {
         // Fill remaining with zeros on error
         while(idx < 130) features[idx++] = 0.0f;
      }
      
      // Pad to 130 if needed
      while(idx < 130) features[idx++] = 0.0f;
      
      return true;
   }
   
   bool IsInitialized() const { return m_initialized; }
   
private:
   //+------------------------------------------------------------------+
   //| Calculate MTF features for LightGBM (10 features: 16-25)        |
   //+------------------------------------------------------------------+
   bool CalculateMTFFeatures(float &features[], int &idx, const int shift)
   {
      // M5 context
      double m5_close = iClose(m_symbol, PERIOD_M5, 1);
      double m5_ema20_buffer[];
      ArraySetAsSeries(m5_ema20_buffer, true);
      if(CopyBuffer(m_ema20_m5_handle, 0, 1, 1, m5_ema20_buffer) <= 0) return false;
      
      features[idx++] = (float)((m5_close > m5_ema20_buffer[0]) ? 1.0 : -1.0);  // M5_trend
      double m5_high = iHigh(m_symbol, PERIOD_M5, 1);
      double m5_low = iLow(m_symbol, PERIOD_M5, 1);
      features[idx++] = (float)((m5_close - m5_low) / (m5_high - m5_low + 0.0001));  // M5_position
      
      // M15 context
      double m15_close = iClose(m_symbol, PERIOD_M15, 1);
      double m15_ema20_buffer[];
      ArraySetAsSeries(m15_ema20_buffer, true);
      if(CopyBuffer(m_ema20_m15_handle, 0, 1, 1, m15_ema20_buffer) <= 0) return false;
      
      features[idx++] = (float)((m15_close > m15_ema20_buffer[0]) ? 1.0 : -1.0);  // M15_trend
      double m15_high = iHigh(m_symbol, PERIOD_M15, 1);
      double m15_low = iLow(m_symbol, PERIOD_M15, 1);
      features[idx++] = (float)((m15_close - m15_low) / (m15_high - m15_low + 0.0001));  // M15_position
      
      // H1 context
      double h1_close = iClose(m_symbol, PERIOD_H1, 1);
      double h1_ema20_buffer[];
      ArraySetAsSeries(h1_ema20_buffer, true);
      if(CopyBuffer(m_ema20_h1_handle, 0, 1, 1, h1_ema20_buffer) <= 0) return false;
      
      features[idx++] = (float)((h1_close > h1_ema20_buffer[0]) ? 1.0 : -1.0);  // H1_trend
      double h1_high = iHigh(m_symbol, PERIOD_H1, 1);
      double h1_low = iLow(m_symbol, PERIOD_H1, 1);
      features[idx++] = (float)((h1_close - h1_low) / (h1_high - h1_low + 0.0001));  // H1_position
      
      // H4 context
      double h4_close = iClose(m_symbol, PERIOD_H4, 1);
      double h4_ema20_buffer[];
      ArraySetAsSeries(h4_ema20_buffer, true);
      if(CopyBuffer(m_ema20_h4_handle, 0, 1, 1, h4_ema20_buffer) <= 0) return false;
      
      features[idx++] = (float)((h4_close > h4_ema20_buffer[0]) ? 1.0 : -1.0);  // H4_trend
      double h4_high = iHigh(m_symbol, PERIOD_H4, 1);
      double h4_low = iLow(m_symbol, PERIOD_H4, 1);
      features[idx++] = (float)((h4_close - h4_low) / (h4_high - h4_low + 0.0001));  // H4_position
      
      // D1 context
      double d1_close = iClose(m_symbol, PERIOD_D1, 1);
      double d1_ema20_buffer[];
      ArraySetAsSeries(d1_ema20_buffer, true);
      if(CopyBuffer(m_ema20_d1_handle, 0, 1, 1, d1_ema20_buffer) <= 0) return false;
      
      features[idx++] = (float)((d1_close > d1_ema20_buffer[0]) ? 1.0 : -1.0);  // D1_trend
      double d1_high = iHigh(m_symbol, PERIOD_D1, 1);
      double d1_low = iLow(m_symbol, PERIOD_D1, 1);
      features[idx++] = (float)((d1_close - d1_low) / (d1_high - d1_low + 0.0001));  // D1_position
      
      return true;
   }
   
   //+------------------------------------------------------------------+
   //| Calculate extended MTF features for Transformer                  |
   //+------------------------------------------------------------------+
   bool CalculateExtendedMTFFeatures(float &features[], int &idx, const int shift)
   {
      // Extended MTF features for each timeframe
      ENUM_TIMEFRAMES tfs[] = {PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1};
      int ema_handles[] = {m_ema20_m5_handle, m_ema20_m15_handle, m_ema20_h1_handle, m_ema20_h4_handle, m_ema20_d1_handle};
      
      for(int t = 0; t < 5; t++)
      {
         ENUM_TIMEFRAMES tf = tfs[t];
         
         // OHLC
         double tf_o = iOpen(m_symbol, tf, 1);
         double tf_h = iHigh(m_symbol, tf, 1);
         double tf_l = iLow(m_symbol, tf, 1);
         double tf_c = iClose(m_symbol, tf, 1);
         
         // Body and range
         features[idx++] = (float)(tf_c - tf_o);
         features[idx++] = (float)(tf_h - tf_l);
         
         // Position
         features[idx++] = (float)((tf_c - tf_l) / (tf_h - tf_l + 0.0001));
         
         // EMA trend
         double ema_buffer[];
         ArraySetAsSeries(ema_buffer, true);
         if(CopyBuffer(ema_handles[t], 0, 1, 1, ema_buffer) <= 0)
            features[idx++] = 0.0f;
         else
            features[idx++] = (float)((tf_c > ema_buffer[0]) ? 1.0 : -1.0);
         
         // Distance from EMA
         if(ArraySize(ema_buffer) > 0)
            features[idx++] = (float)((tf_c - ema_buffer[0]) / (ema_buffer[0] + 0.0001));
         else
            features[idx++] = 0.0f;
      }
      
      return true;
   }
};
