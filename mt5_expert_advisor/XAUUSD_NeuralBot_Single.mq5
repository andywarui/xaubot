//+------------------------------------------------------------------+
//|                                    XAUUSD_NeuralBot_M1.mq5       |
//|                        M1-based execution with ONNX inference    |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Neural Bot M1"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input double RiskPercent = 0.5;
input double ConfidenceThreshold = 0.60;
input int MaxTradesPerDay = 5;
input double MaxDailyLoss = 4.0;
input double StopLossUSD = 4.0;         // Stop loss in USD (e.g., 4.0 = $4 price move)
input double TakeProfitUSD = 8.0;       // Take profit in USD (e.g., 8.0 = $8 price move)
input double MaxMarginPercent = 50.0;   // Max % of free margin to use per trade
input bool EnableFeatureLog = false;
input string FeatureLogFile = "feature_log.csv";
input bool EnablePredictionLog = false;
input string PredictionLogFile = "prediction_log.csv";
input bool EnableOnnxDebugLogs = false;  // ONNX debug logs (development only)

//--- Hybrid Validation Filters (Research-based optimization)
input group "=== Hybrid Validation Filters ==="
input bool EnableHybridValidation = true;   // Enable ML + Technical validation
input double MaxSpreadUSD = 2.0;            // Max spread in USD
input double RSI_OverboughtLevel = 70.0;    // RSI overbought threshold
input double RSI_OversoldLevel = 30.0;      // RSI oversold threshold
input double ATR_MinLevel = 1.5;            // Min ATR (avoid low volatility)
input double ATR_MaxLevel = 8.0;            // Max ATR (avoid extreme volatility)
input double ADX_MinStrength = 20.0;        // Min ADX (trend strength)
input bool RequireMTFAlignment = true;      // Require multi-timeframe EMA alignment

CTrade trade;
datetime lastM1BarTime = 0;
int tradesOpenedToday = 0;
datetime lastResetDate = 0;
double startingEquity = 0;

string FEATURE_NAMES[26] = {
   "body","body_abs","candle_range","close_position",
   "return_1","return_5","return_15","return_60",
   "tr","atr_14","rsi_14","ema_10","ema_20","ema_50",
   "hour_sin","hour_cos","M5_trend","M5_position",
   "M15_trend","M15_position","H1_trend","H1_position",
   "H4_trend","H4_position","D1_trend","D1_position"
};

// ONNX model
long g_onnxHandle = INVALID_HANDLE;
bool g_modelReady = false;

// Indicator handles (create once; release on deinit)
int g_atr14_m1_handle = INVALID_HANDLE;
int g_rsi14_m1_handle = INVALID_HANDLE;
int g_ema10_m1_handle = INVALID_HANDLE;
int g_ema20_m1_handle = INVALID_HANDLE;
int g_ema50_m1_handle = INVALID_HANDLE;

int g_ema20_m5_handle  = INVALID_HANDLE;
int g_ema20_m15_handle = INVALID_HANDLE;
int g_ema20_h1_handle  = INVALID_HANDLE;
int g_ema20_h4_handle  = INVALID_HANDLE;
int g_ema20_d1_handle  = INVALID_HANDLE;

// Hybrid validation indicator handles
int g_macd_handle = INVALID_HANDLE;
int g_adx_handle = INVALID_HANDLE;

long LoadOnnxWithFallback(const string model_file, const uint flags)
{
   // Strategy Tester sometimes runs in a different data folder (agent sandbox),
   // so log paths and try a few common relative/absolute variants.
   const string candidates[] = {
      model_file,
      "Files\\" + model_file,
      "MQL5\\Files\\" + model_file,
      TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + model_file,
      TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + model_file
   };

   PrintFormat("ONNX load: Terminal data path=%s", TerminalInfoString(TERMINAL_DATA_PATH));
   PrintFormat("ONNX load: Terminal common path=%s", TerminalInfoString(TERMINAL_COMMONDATA_PATH));
   PrintFormat("ONNX load: FileIsExist('%s')=%s", model_file, FileIsExist(model_file) ? "true" : "false");

   for(int i = 0; i < ArraySize(candidates); i++)
   {
      string path = candidates[i];
      ResetLastError();
      long h = OnnxCreate(path, flags);
      if(h != INVALID_HANDLE)
      {
         if(path != model_file)
            PrintFormat("ONNX load: succeeded using '%s'", path);
         return h;
      }
      PrintFormat("ONNX load: failed using '%s' (err=%d)", path, GetLastError());
   }

   return INVALID_HANDLE;
}

bool InitIndicators()
{
   g_atr14_m1_handle = iATR(_Symbol, PERIOD_M1, 14);
   g_rsi14_m1_handle = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
   g_ema10_m1_handle = iMA(_Symbol, PERIOD_M1, 10, 0, MODE_EMA, PRICE_CLOSE);
   g_ema20_m1_handle = iMA(_Symbol, PERIOD_M1, 20, 0, MODE_EMA, PRICE_CLOSE);
   g_ema50_m1_handle = iMA(_Symbol, PERIOD_M1, 50, 0, MODE_EMA, PRICE_CLOSE);

   g_ema20_m5_handle  = iMA(_Symbol, PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE);
   g_ema20_m15_handle = iMA(_Symbol, PERIOD_M15, 20, 0, MODE_EMA, PRICE_CLOSE);
   g_ema20_h1_handle  = iMA(_Symbol, PERIOD_H1, 20, 0, MODE_EMA, PRICE_CLOSE);
   g_ema20_h4_handle  = iMA(_Symbol, PERIOD_H4, 20, 0, MODE_EMA, PRICE_CLOSE);
   g_ema20_d1_handle  = iMA(_Symbol, PERIOD_D1, 20, 0, MODE_EMA, PRICE_CLOSE);

   // Hybrid validation indicators (M15 timeframe for filtering)
   g_macd_handle = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
   g_adx_handle = iADX(_Symbol, PERIOD_M15, 14);

   if(g_atr14_m1_handle == INVALID_HANDLE || g_rsi14_m1_handle == INVALID_HANDLE ||
      g_ema10_m1_handle == INVALID_HANDLE || g_ema20_m1_handle == INVALID_HANDLE || g_ema50_m1_handle == INVALID_HANDLE ||
      g_ema20_m5_handle == INVALID_HANDLE || g_ema20_m15_handle == INVALID_HANDLE || g_ema20_h1_handle == INVALID_HANDLE ||
      g_ema20_h4_handle == INVALID_HANDLE || g_ema20_d1_handle == INVALID_HANDLE ||
      g_macd_handle == INVALID_HANDLE || g_adx_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create one or more indicator handles");
      return false;
   }

   Print("✓ All indicators initialized successfully (including hybrid validation)");
   return true;
}

//+------------------------------------------------------------------+
int OnInit()
{
   Print("XAUUSD Neural Bot M1 - Initializing...");
   
   // Load ONNX model from MQL5/Files/ folder
   string model_file = "lightgbm_xauusd.onnx";
   uint flags = EnableOnnxDebugLogs ? ONNX_DEBUG_LOGS : ONNX_DEFAULT;
   g_onnxHandle = LoadOnnxWithFallback(model_file, flags);
   
   if(g_onnxHandle == INVALID_HANDLE)
   {
      int error = GetLastError();
      Print("ERROR: Failed to load ONNX model: ", model_file);
      Print("Error code: ", error);
      Print("Make sure file is in the running terminal's MQL5/Files/ and (for Strategy Tester) the agent sandbox MQL5/Files/");
      return(INIT_FAILED);
   }

   if(!InitIndicators())
   {
      Print("ERROR: Indicator initialization failed");
      OnnxRelease(g_onnxHandle);
      g_onnxHandle = INVALID_HANDLE;
      return(INIT_FAILED);
   }
   
   // Set input shape [1, 26] - fixed batch size for consistent inference
   ulong input_shape[] = {1, 26};
   if(!OnnxSetInputShape(g_onnxHandle, 0, input_shape))
   {
      int error = GetLastError();
      Print("ERROR: Failed to set ONNX input shape [1, 26]");
      Print("Error code: ", error);
      OnnxRelease(g_onnxHandle);
      g_onnxHandle = INVALID_HANDLE;
      return(INIT_FAILED);
   }
   
   // Explicitly set output shapes to match buffer dimensions
   // Output 0: label [1] - int64 scalar (model's predicted class)
   ulong label_shape[] = {1};
   if(!OnnxSetOutputShape(g_onnxHandle, 0, label_shape))
   {
      int error = GetLastError();
      Print("ERROR: Failed to set ONNX output[0] shape [1], code=", error);
      OnnxRelease(g_onnxHandle);
      g_onnxHandle = INVALID_HANDLE;
      return(INIT_FAILED);
   }
   
   // Output 1: probabilities [1,3] - SHORT, HOLD, LONG probabilities
   ulong probs_shape[] = {1, 3};
   if(!OnnxSetOutputShape(g_onnxHandle, 1, probs_shape))
   {
      int error = GetLastError();
      Print("ERROR: Failed to set ONNX output[1] shape [1,3], code=", error);
      OnnxRelease(g_onnxHandle);
      g_onnxHandle = INVALID_HANDLE;
      return(INIT_FAILED);
   }
   
   g_modelReady = true;
   startingEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   lastM1BarTime = iTime(_Symbol, PERIOD_M1, 0);
   
   Print("=== XAUUSD Neural Bot M1 - Initialized ===");
   Print("ONNX Model: ", model_file, " - Loaded successfully");
   Print("Input shape: [1, 26] features");
   Print("Output 0: label [1] (int64)");
   Print("Output 1: probabilities [1, 3] (float)");
   Print("Execution: M5 chart, trades on M1 bars");
   Print("Debug logs: ", EnableOnnxDebugLogs ? "ENABLED" : "DISABLED");
   Print("Risk: ", RiskPercent, "% | Confidence: ", ConfidenceThreshold*100, "%");
   PrintFormat("SL: $%.2f | TP: $%.2f | Max Margin: %.0f%%", StopLossUSD, TakeProfitUSD, MaxMarginPercent);
   Print("Ready to trade. Waiting for M1 bars...");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_onnxHandle != INVALID_HANDLE)
   {
      OnnxRelease(g_onnxHandle);
      g_onnxHandle = INVALID_HANDLE;
      Print("ONNX model released");
   }

   if(g_atr14_m1_handle != INVALID_HANDLE) { IndicatorRelease(g_atr14_m1_handle); g_atr14_m1_handle = INVALID_HANDLE; }
   if(g_rsi14_m1_handle != INVALID_HANDLE) { IndicatorRelease(g_rsi14_m1_handle); g_rsi14_m1_handle = INVALID_HANDLE; }
   if(g_ema10_m1_handle != INVALID_HANDLE) { IndicatorRelease(g_ema10_m1_handle); g_ema10_m1_handle = INVALID_HANDLE; }
   if(g_ema20_m1_handle != INVALID_HANDLE) { IndicatorRelease(g_ema20_m1_handle); g_ema20_m1_handle = INVALID_HANDLE; }
   if(g_ema50_m1_handle != INVALID_HANDLE) { IndicatorRelease(g_ema50_m1_handle); g_ema50_m1_handle = INVALID_HANDLE; }

   if(g_ema20_m5_handle != INVALID_HANDLE)  { IndicatorRelease(g_ema20_m5_handle);  g_ema20_m5_handle = INVALID_HANDLE; }
   if(g_ema20_m15_handle != INVALID_HANDLE) { IndicatorRelease(g_ema20_m15_handle); g_ema20_m15_handle = INVALID_HANDLE; }
   if(g_ema20_h1_handle != INVALID_HANDLE)  { IndicatorRelease(g_ema20_h1_handle);  g_ema20_h1_handle = INVALID_HANDLE; }
   if(g_ema20_h4_handle != INVALID_HANDLE)  { IndicatorRelease(g_ema20_h4_handle);  g_ema20_h4_handle = INVALID_HANDLE; }
   if(g_ema20_d1_handle != INVALID_HANDLE)  { IndicatorRelease(g_ema20_d1_handle);  g_ema20_d1_handle = INVALID_HANDLE; }

   // Release hybrid validation indicators
   if(g_macd_handle != INVALID_HANDLE) { IndicatorRelease(g_macd_handle); g_macd_handle = INVALID_HANDLE; }
   if(g_adx_handle != INVALID_HANDLE) { IndicatorRelease(g_adx_handle); g_adx_handle = INVALID_HANDLE; }

   Print("All indicators released");
}

//+------------------------------------------------------------------+
bool IsNewM1Bar()
{
   datetime currentM1Time = iTime(_Symbol, PERIOD_M1, 0);
   if(currentM1Time != lastM1BarTime)
   {
      lastM1BarTime = currentM1Time;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
void ResetDailyCounters()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime currentDate = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
   
   if(currentDate != lastResetDate)
   {
      tradesOpenedToday = 0;
      lastResetDate = currentDate;
   }
}

//+------------------------------------------------------------------+
bool CheckRiskLimits()
{
   if(tradesOpenedToday >= MaxTradesPerDay) return false;
   
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyLoss = (startingEquity - currentEquity) / startingEquity * 100;
   if(dailyLoss >= MaxDailyLoss) return false;
   
   return true;
}

//+------------------------------------------------------------------+
bool IsInSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return (dt.hour >= 12 && dt.hour < 17);
}

//+------------------------------------------------------------------+
//| Calculate M1 Features - MUST MATCH Python Training Preprocessing |
//| Feature order defined in config/features_order.json              |
//| NO normalization/scaling applied here (model trained on raw)     |
//+------------------------------------------------------------------+
bool CalculateM1Features(float &features[], const int shift)
{
   ArrayResize(features, 26);
   
   // Get M1 OHLC
   double o = iOpen(_Symbol, PERIOD_M1, shift);
   double h = iHigh(_Symbol, PERIOD_M1, shift);
   double l = iLow(_Symbol, PERIOD_M1, shift);
   double c = iClose(_Symbol, PERIOD_M1, shift);
   
   if(o <= 0 || h <= 0 || l <= 0 || c <= 0) return false;
   
   int idx = 0;
   
   // 0-3: Price features
   features[idx++] = (float)(c - o);
   features[idx++] = (float)MathAbs(c - o);
   features[idx++] = (float)(h - l);
   features[idx++] = (float)((c - l) / (h - l + 0.0001));
   
   // 4-7: Returns
   double c1 = iClose(_Symbol, PERIOD_M1, shift + 1);
   double c5 = iClose(_Symbol, PERIOD_M1, shift + 5);
   double c15 = iClose(_Symbol, PERIOD_M1, shift + 15);
   double c60 = iClose(_Symbol, PERIOD_M1, shift + 60);
   
   features[idx++] = (float)((c - c1) / (c1 + 0.0001));
   features[idx++] = (float)((c - c5) / (c5 + 0.0001));
   features[idx++] = (float)((c - c15) / (c15 + 0.0001));
   features[idx++] = (float)((c - c60) / (c60 + 0.0001));
   
   // 8-9: ATR
   double prev_close = iClose(_Symbol, PERIOD_M1, shift + 1);
   double tr = MathMax(h - l, MathMax(MathAbs(h - prev_close), MathAbs(l - prev_close)));
   features[idx++] = (float)tr;
   
   double atr14_buffer[];
   ArraySetAsSeries(atr14_buffer, true);
   if(CopyBuffer(g_atr14_m1_handle, 0, shift, 1, atr14_buffer) <= 0) return false;
   features[idx++] = (float)atr14_buffer[0];
   
   // 10: RSI
   double rsi14_buffer[];
   ArraySetAsSeries(rsi14_buffer, true);
   if(CopyBuffer(g_rsi14_m1_handle, 0, shift, 1, rsi14_buffer) <= 0) return false;
   features[idx++] = (float)rsi14_buffer[0];
   
   // 11-13: EMAs
   double ema10_buffer[], ema20_buffer[], ema50_buffer[];
   ArraySetAsSeries(ema10_buffer, true);
   ArraySetAsSeries(ema20_buffer, true);
   ArraySetAsSeries(ema50_buffer, true);

   if(CopyBuffer(g_ema10_m1_handle, 0, shift, 1, ema10_buffer) <= 0) return false;
   if(CopyBuffer(g_ema20_m1_handle, 0, shift, 1, ema20_buffer) <= 0) return false;
   if(CopyBuffer(g_ema50_m1_handle, 0, shift, 1, ema50_buffer) <= 0) return false;
   
   features[idx++] = (float)ema10_buffer[0];
   features[idx++] = (float)ema20_buffer[0];
   features[idx++] = (float)ema50_buffer[0];
   
   // 14-15: Time features
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   features[idx++] = (float)MathSin(2 * M_PI * dt.hour / 24);
   features[idx++] = (float)MathCos(2 * M_PI * dt.hour / 24);
   
   // 16-17: M5 context
   double m5_close = iClose(_Symbol, PERIOD_M5, 1);
   double m5_ema20_buffer[];
   ArraySetAsSeries(m5_ema20_buffer, true);
   if(CopyBuffer(g_ema20_m5_handle, 0, 1, 1, m5_ema20_buffer) <= 0) return false;
   
   features[idx++] = (float)((m5_close > m5_ema20_buffer[0]) ? 1.0 : -1.0);
   
   double m5_high = iHigh(_Symbol, PERIOD_M5, 1);
   double m5_low = iLow(_Symbol, PERIOD_M5, 1);
   features[idx++] = (float)((m5_close - m5_low) / (m5_high - m5_low + 0.0001));
   
   // 18-19: M15 context
   double m15_close = iClose(_Symbol, PERIOD_M15, 1);
   double m15_ema20_buffer[];
   ArraySetAsSeries(m15_ema20_buffer, true);
   if(CopyBuffer(g_ema20_m15_handle, 0, 1, 1, m15_ema20_buffer) <= 0) return false;
   
   features[idx++] = (float)((m15_close > m15_ema20_buffer[0]) ? 1.0 : -1.0);
   
   double m15_high = iHigh(_Symbol, PERIOD_M15, 1);
   double m15_low = iLow(_Symbol, PERIOD_M15, 1);
   features[idx++] = (float)((m15_close - m15_low) / (m15_high - m15_low + 0.0001));
   
   // 20-21: H1 context
   double h1_close = iClose(_Symbol, PERIOD_H1, 1);
   double h1_ema20_buffer[];
   ArraySetAsSeries(h1_ema20_buffer, true);
   if(CopyBuffer(g_ema20_h1_handle, 0, 1, 1, h1_ema20_buffer) <= 0) return false;
   
   features[idx++] = (float)((h1_close > h1_ema20_buffer[0]) ? 1.0 : -1.0);
   
   double h1_high = iHigh(_Symbol, PERIOD_H1, 1);
   double h1_low = iLow(_Symbol, PERIOD_H1, 1);
   features[idx++] = (float)((h1_close - h1_low) / (h1_high - h1_low + 0.0001));
   
   // 22-23: H4 context
   double h4_close = iClose(_Symbol, PERIOD_H4, 1);
   double h4_ema20_buffer[];
   ArraySetAsSeries(h4_ema20_buffer, true);
   if(CopyBuffer(g_ema20_h4_handle, 0, 1, 1, h4_ema20_buffer) <= 0) return false;
   
   features[idx++] = (float)((h4_close > h4_ema20_buffer[0]) ? 1.0 : -1.0);
   
   double h4_high = iHigh(_Symbol, PERIOD_H4, 1);
   double h4_low = iLow(_Symbol, PERIOD_H4, 1);
   features[idx++] = (float)((h4_close - h4_low) / (h4_high - h4_low + 0.0001));
   
   // 24-25: D1 context
   double d1_close = iClose(_Symbol, PERIOD_D1, 1);
   double d1_ema20_buffer[];
   ArraySetAsSeries(d1_ema20_buffer, true);
   if(CopyBuffer(g_ema20_d1_handle, 0, 1, 1, d1_ema20_buffer) <= 0) return false;
   
   features[idx++] = (float)((d1_close > d1_ema20_buffer[0]) ? 1.0 : -1.0);
   
   double d1_high = iHigh(_Symbol, PERIOD_D1, 1);
   double d1_low = iLow(_Symbol, PERIOD_D1, 1);
   features[idx++] = (float)((d1_close - d1_low) / (d1_high - d1_low + 0.0001));
   
   return true;
}

//+------------------------------------------------------------------+
bool LogFeatures(datetime barTime, float &features[])
{
   int handle = FileOpen(FeatureLogFile, FILE_CSV | FILE_READ | FILE_WRITE | FILE_ANSI, ';');
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open feature log file: ", FeatureLogFile);
      return false;
   }

   FileSeek(handle, 0, SEEK_END);
   if(FileTell(handle) == 0)
   {
      FileWrite(handle,
         "time",
         FEATURE_NAMES[0], FEATURE_NAMES[1], FEATURE_NAMES[2], FEATURE_NAMES[3], FEATURE_NAMES[4], FEATURE_NAMES[5], FEATURE_NAMES[6], FEATURE_NAMES[7],
         FEATURE_NAMES[8], FEATURE_NAMES[9], FEATURE_NAMES[10], FEATURE_NAMES[11], FEATURE_NAMES[12], FEATURE_NAMES[13], FEATURE_NAMES[14], FEATURE_NAMES[15],
         FEATURE_NAMES[16], FEATURE_NAMES[17], FEATURE_NAMES[18], FEATURE_NAMES[19], FEATURE_NAMES[20], FEATURE_NAMES[21], FEATURE_NAMES[22], FEATURE_NAMES[23],
         FEATURE_NAMES[24], FEATURE_NAMES[25]
      );
   }

   string ts = TimeToString(barTime, TIME_DATE | TIME_MINUTES | TIME_SECONDS);
   FileWrite(handle,
      ts,
      features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7],
      features[8], features[9], features[10], features[11], features[12], features[13], features[14], features[15],
      features[16], features[17], features[18], features[19], features[20], features[21], features[22], features[23],
      features[24], features[25]
   );

   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
bool LogPrediction(datetime barTime, double p_short, double p_hold, double p_long, int best, double best_p)
{
   int handle = FileOpen(PredictionLogFile, FILE_CSV | FILE_READ | FILE_WRITE | FILE_ANSI, ';');
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open prediction log file: ", PredictionLogFile);
      return false;
   }

   FileSeek(handle, 0, SEEK_END);
   if(FileTell(handle) == 0)
   {
      FileWrite(handle, "time", "p_short", "p_hold", "p_long", "best_class", "best_prob");
   }

   string ts = TimeToString(barTime, TIME_DATE | TIME_MINUTES | TIME_SECONDS);
   FileWrite(handle, ts, p_short, p_hold, p_long, best, best_p);
   FileClose(handle);
   return true;
}

//+------------------------------------------------------------------+
int PredictSignal(const datetime barTime, double &out_confidence)
{
   out_confidence = 0.0;
   
   if(!g_modelReady || g_onnxHandle == INVALID_HANDLE)
      return 1; // HOLD
   
   // Calculate features
   float features[];
   // Use the last CLOSED bar (shift=1) to avoid look-ahead bias.
   if(!CalculateM1Features(features, 1))
   {
      Print("ERROR: Failed to calculate features");
      return 1;
   }

   if(EnableFeatureLog)
      LogFeatures(barTime, features);
   
   // Run ONNX inference with preallocated buffers (performance optimization)
   // Note: ONNX model has 2 outputs:
   //   Output 0: label (int64[1]) - predicted class index
   //   Output 1: probabilities (float[1,3]) - SHORT, HOLD, LONG probabilities
   static float input_data[26];     // 26 features
   static long  output_label[1];    // label output (int64 -> long in MQL5)
   static float output_probs[3];    // probabilities: [p_short, p_hold, p_long]
   
   // Copy features to input buffer
   ArrayCopy(input_data, features, 0, 0, 26);
   
   // Execute ONNX inference: 1 input + 2 outputs = 3 data parameters
   // Using ONNX_DEFAULT to allow type conversions if needed
   if(!OnnxRun(g_onnxHandle, ONNX_DEFAULT, input_data, output_label, output_probs))
   {
      int error = GetLastError();
      Print("ERROR: ONNX inference failed - Error code: ", error);
      PrintFormat("  Model label output: %d", output_label[0]);
      Print("Returning HOLD signal (safe mode)");
      return 1;  // Return HOLD on inference failure
   }
   
   // Extract probabilities from output buffer
   // Index mapping: 0=SHORT, 1=HOLD, 2=LONG
   double p_short = output_probs[0];
   double p_hold  = output_probs[1];
   double p_long  = output_probs[2];
   
   // Model's predicted class (can be used for validation)
   int model_class = (int)output_label[0];  // 0=SHORT, 1=HOLD, 2=LONG
   
   // Validation: Check if probabilities sum to ~1.0 and are in valid range [0, 1]
   double prob_sum = p_short + p_hold + p_long;
   if(prob_sum < 0.99 || prob_sum > 1.01 || 
      p_short < 0.0 || p_short > 1.0 ||
      p_hold < 0.0 || p_hold > 1.0 ||
      p_long < 0.0 || p_long > 1.0)
   {
      Print("WARNING: Invalid probabilities detected!");
      PrintFormat("  Sum = %.6f (expected ~1.0)", prob_sum);
      PrintFormat("  Model class: %d, Raw probs: [%.6f, %.6f, %.6f]", 
            model_class, output_probs[0], output_probs[1], output_probs[2]);
      Print("Returning HOLD signal (safe mode)");
      return 1;  // Return HOLD on invalid probabilities
   }
   
   // Find best prediction
   int best = 0;
   double best_p = p_short;
   
   if(p_hold > best_p) { best_p = p_hold; best = 1; }
   if(p_long > best_p) { best_p = p_long; best = 2; }
   
   if(EnablePredictionLog)
      LogPrediction(barTime, p_short, p_hold, p_long, best, best_p);

   // Apply confidence threshold
   if(best_p < ConfidenceThreshold)
      return 1; // HOLD
   
   out_confidence = best_p;
   return best; // 0=SHORT, 1=HOLD, 2=LONG
}

//+------------------------------------------------------------------+
double CalculateLotSize(ENUM_ORDER_TYPE orderType)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskAmount = equity * RiskPercent / 100.0;
   
   // For XAUUSD: Calculate risk per lot for the configured stop loss
   // XAUUSD typically: 1 lot = 100 oz, tick size = 0.01, tick value = $1
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double stopLossPoints = StopLossUSD;  // Use configurable SL in USD
   
   // Risk per lot = (stopLossPoints / tickSize) * tickValue
   double riskPerLot = (stopLossPoints / tickSize) * tickValue;
   
   // Prevent division by zero
   if(riskPerLot <= 0)
   {
      Print("WARNING: Invalid risk per lot calculation");
      return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   }
   
   double lotSize = riskAmount / riskPerLot;
   
   // Round to lot step
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   
   // Clamp to broker limits
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   
   if(lotSize < minLot) lotSize = minLot;
   if(lotSize > maxLot) lotSize = maxLot;
   
   // Final margin check - reduce lot size if margin insufficient
   double margin_required = 0;
   double price = (orderType == ORDER_TYPE_BUY) ? 
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                  SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   if(!OrderCalcMargin(orderType, _Symbol, lotSize, price, margin_required))
   {
      Print("WARNING: OrderCalcMargin failed");
      return minLot;
   }
   
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   
   // If margin required exceeds MaxMarginPercent of free margin, scale down
   double maxMarginRatio = MaxMarginPercent / 100.0;
   if(margin_required > freeMargin * maxMarginRatio)
   {
      double scale = (freeMargin * maxMarginRatio) / margin_required;
      lotSize = MathFloor((lotSize * scale) / lotStep) * lotStep;
      if(lotSize < minLot) lotSize = 0;  // Signal: cannot afford even min lot
   }
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| Hybrid Validation: ML + Technical Indicators (Research-based)   |
//| Filters false signals using multi-layer validation              |
//+------------------------------------------------------------------+
bool ValidateLongSignal(double ml_confidence)
{
   if(!EnableHybridValidation)
      return true;  // Skip validation if disabled

   // 1. Spread filter - avoid high transaction costs
   double spread_usd = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                        SymbolInfoDouble(_Symbol, SYMBOL_BID));

   if(spread_usd > MaxSpreadUSD)
   {
      if(EnablePredictionLog)
         Print("FILTER REJECT [LONG]: Spread too high: $", DoubleToString(spread_usd, 2),
               " > $", DoubleToString(MaxSpreadUSD, 2));
      return false;
   }

   // 2. RSI filter - avoid overbought conditions
   double rsi[];
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(g_rsi14_m1_handle, 0, 0, 1, rsi) > 0)
   {
      if(rsi[0] > RSI_OverboughtLevel)
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [LONG]: RSI overbought: ", DoubleToString(rsi[0], 1),
                  " > ", DoubleToString(RSI_OverboughtLevel, 1));
         return false;
      }
   }

   // 3. MACD alignment - bullish confirmation
   double macd_main[], macd_signal[];
   ArraySetAsSeries(macd_main, true);
   ArraySetAsSeries(macd_signal, true);

   if(CopyBuffer(g_macd_handle, 0, 0, 1, macd_main) > 0 &&
      CopyBuffer(g_macd_handle, 1, 0, 1, macd_signal) > 0)
   {
      if(macd_main[0] < macd_signal[0])
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [LONG]: MACD bearish (main: ", DoubleToString(macd_main[0], 2),
                  " < signal: ", DoubleToString(macd_signal[0], 2), ")");
         return false;
      }
   }

   // 4. ADX trend strength - ensure trending market
   double adx[];
   ArraySetAsSeries(adx, true);
   if(CopyBuffer(g_adx_handle, 0, 0, 1, adx) > 0)
   {
      if(adx[0] < ADX_MinStrength)
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [LONG]: ADX too weak (ranging): ", DoubleToString(adx[0], 1),
                  " < ", DoubleToString(ADX_MinStrength, 1));
         return false;
      }
   }

   // 5. ATR volatility filter - sweet spot for trading
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(g_atr14_m1_handle, 0, 0, 1, atr) > 0)
   {
      if(atr[0] < ATR_MinLevel || atr[0] > ATR_MaxLevel)
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [LONG]: ATR outside range: ", DoubleToString(atr[0], 2),
                  " (range: ", DoubleToString(ATR_MinLevel, 2), "-", DoubleToString(ATR_MaxLevel, 2), ")");
         return false;
      }
   }

   // 6. Multi-timeframe EMA alignment - higher TF confirmation
   if(RequireMTFAlignment)
   {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double ema20_m15[], ema20_h1[];
      ArraySetAsSeries(ema20_m15, true);
      ArraySetAsSeries(ema20_h1, true);

      if(CopyBuffer(g_ema20_m15_handle, 0, 0, 1, ema20_m15) > 0 &&
         CopyBuffer(g_ema20_h1_handle, 0, 0, 1, ema20_h1) > 0)
      {
         if(price < ema20_m15[0] || price < ema20_h1[0])
         {
            if(EnablePredictionLog)
               Print("FILTER REJECT [LONG]: Price below higher TF EMAs (M15: ",
                     DoubleToString(ema20_m15[0], 2), ", H1: ", DoubleToString(ema20_h1[0], 2), ")");
            return false;
         }
      }
   }

   if(EnablePredictionLog)
      Print("✓ LONG signal passed all hybrid validation filters (confidence: ",
            DoubleToString(ml_confidence, 3), ")");

   return true;
}

//+------------------------------------------------------------------+
bool ValidateShortSignal(double ml_confidence)
{
   if(!EnableHybridValidation)
      return true;  // Skip validation if disabled

   // 1. Spread filter - avoid high transaction costs
   double spread_usd = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                        SymbolInfoDouble(_Symbol, SYMBOL_BID));

   if(spread_usd > MaxSpreadUSD)
   {
      if(EnablePredictionLog)
         Print("FILTER REJECT [SHORT]: Spread too high: $", DoubleToString(spread_usd, 2),
               " > $", DoubleToString(MaxSpreadUSD, 2));
      return false;
   }

   // 2. RSI filter - avoid oversold conditions
   double rsi[];
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(g_rsi14_m1_handle, 0, 0, 1, rsi) > 0)
   {
      if(rsi[0] < RSI_OversoldLevel)
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [SHORT]: RSI oversold: ", DoubleToString(rsi[0], 1),
                  " < ", DoubleToString(RSI_OversoldLevel, 1));
         return false;
      }
   }

   // 3. MACD alignment - bearish confirmation
   double macd_main[], macd_signal[];
   ArraySetAsSeries(macd_main, true);
   ArraySetAsSeries(macd_signal, true);

   if(CopyBuffer(g_macd_handle, 0, 0, 1, macd_main) > 0 &&
      CopyBuffer(g_macd_handle, 1, 0, 1, macd_signal) > 0)
   {
      if(macd_main[0] > macd_signal[0])
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [SHORT]: MACD bullish (main: ", DoubleToString(macd_main[0], 2),
                  " > signal: ", DoubleToString(macd_signal[0], 2), ")");
         return false;
      }
   }

   // 4. ADX trend strength - ensure trending market
   double adx[];
   ArraySetAsSeries(adx, true);
   if(CopyBuffer(g_adx_handle, 0, 0, 1, adx) > 0)
   {
      if(adx[0] < ADX_MinStrength)
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [SHORT]: ADX too weak (ranging): ", DoubleToString(adx[0], 1),
                  " < ", DoubleToString(ADX_MinStrength, 1));
         return false;
      }
   }

   // 5. ATR volatility filter - sweet spot for trading
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(g_atr14_m1_handle, 0, 0, 1, atr) > 0)
   {
      if(atr[0] < ATR_MinLevel || atr[0] > ATR_MaxLevel)
      {
         if(EnablePredictionLog)
            Print("FILTER REJECT [SHORT]: ATR outside range: ", DoubleToString(atr[0], 2),
                  " (range: ", DoubleToString(ATR_MinLevel, 2), "-", DoubleToString(ATR_MaxLevel, 2), ")");
         return false;
      }
   }

   // 6. Multi-timeframe EMA alignment - higher TF confirmation
   if(RequireMTFAlignment)
   {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ema20_m15[], ema20_h1[];
      ArraySetAsSeries(ema20_m15, true);
      ArraySetAsSeries(ema20_h1, true);

      if(CopyBuffer(g_ema20_m15_handle, 0, 0, 1, ema20_m15) > 0 &&
         CopyBuffer(g_ema20_h1_handle, 0, 0, 1, ema20_h1) > 0)
      {
         if(price > ema20_m15[0] || price > ema20_h1[0])
         {
            if(EnablePredictionLog)
               Print("FILTER REJECT [SHORT]: Price above higher TF EMAs (M15: ",
                     DoubleToString(ema20_m15[0], 2), ", H1: ", DoubleToString(ema20_h1[0], 2), ")");
            return false;
         }
      }
   }

   if(EnablePredictionLog)
      Print("✓ SHORT signal passed all hybrid validation filters (confidence: ",
            DoubleToString(ml_confidence, 3), ")");

   return true;
}

//+------------------------------------------------------------------+
void OnTick()
{
   ResetDailyCounters();
   
   // Only act on new M1 bar
   if(!IsNewM1Bar())
      return;
   
   // Check session and risk limits
   if(!IsInSession() || !CheckRiskLimits())
      return;

   // We act at the OPEN of the new bar, but features are computed from the last CLOSED bar.
   datetime featureBarTime = iTime(_Symbol, PERIOD_M1, 1);
   
   // Get prediction
   double confidence;
   int signal = PredictSignal(featureBarTime, confidence);
   
   // Manage existing position
   if(PositionSelect(_Symbol))
   {
      // TODO: Implement trailing stop
      return;
   }
   
   // Open new position
   if(signal == 2) // LONG
   {
      // Hybrid validation: Check ML signal against technical indicators
      if(!ValidateLongSignal(confidence))
      {
         Print("ML LONG signal rejected by hybrid validation filters");
         return;
      }

      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      // Use configurable SL/TP in USD
      double sl = price - StopLossUSD;
      double tp = price + TakeProfitUSD;
      double lots = CalculateLotSize(ORDER_TYPE_BUY);

      // Skip if lot size is 0 (insufficient margin)
      if(lots <= 0)
      {
         Print("SKIP LONG: Insufficient margin for minimum lot size");
         return;
      }
      
      // Final margin verification before order
      double margin_required = 0;
      if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lots, price, margin_required))
      {
         Print("ERROR: Cannot calculate margin for BUY order");
         return;
      }
      
      double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      if(margin_required > freeMargin)
      {
         PrintFormat("SKIP LONG: Margin required (%.2f) exceeds free margin (%.2f)", 
                     margin_required, freeMargin);
         return;
      }
      
      if(trade.Buy(lots, _Symbol, price, sl, tp, StringFormat("M1 LONG %.2f", confidence)))
      {
         tradesOpenedToday++;
         PrintFormat("M1 LONG: %.2f lots at %.5f (conf: %.2f, margin: %.2f)", 
                     lots, price, confidence, margin_required);
      }
   }
   else if(signal == 0) // SHORT
   {
      // Hybrid validation: Check ML signal against technical indicators
      if(!ValidateShortSignal(confidence))
      {
         Print("ML SHORT signal rejected by hybrid validation filters");
         return;
      }

      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      // Use configurable SL/TP in USD
      double sl = price + StopLossUSD;
      double tp = price - TakeProfitUSD;
      double lots = CalculateLotSize(ORDER_TYPE_SELL);

      // Skip if lot size is 0 (insufficient margin)
      if(lots <= 0)
      {
         Print("SKIP SHORT: Insufficient margin for minimum lot size");
         return;
      }
      
      // Final margin verification before order
      double margin_required = 0;
      if(!OrderCalcMargin(ORDER_TYPE_SELL, _Symbol, lots, price, margin_required))
      {
         Print("ERROR: Cannot calculate margin for SELL order");
         return;
      }
      
      double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      if(margin_required > freeMargin)
      {
         PrintFormat("SKIP SHORT: Margin required (%.2f) exceeds free margin (%.2f)", 
                     margin_required, freeMargin);
         return;
      }
      
      if(trade.Sell(lots, _Symbol, price, sl, tp, StringFormat("M1 SHORT %.2f", confidence)))
      {
         tradesOpenedToday++;
         PrintFormat("M1 SHORT: %.2f lots at %.5f (conf: %.2f, margin: %.2f)", 
                     lots, price, confidence, margin_required);
      }
   }
}
//+------------------------------------------------------------------+
