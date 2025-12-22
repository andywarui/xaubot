//+------------------------------------------------------------------+
//|                              XAUUSD_NeuralBot_Ensemble.mq5      |
//|                    Ensemble: LightGBM + Transformer + Hybrid    |
//|                           Research-backed optimization           |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Neural Bot Ensemble"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

//--- Core Trading Parameters
input group "=== Core Trading Parameters ==="
input double RiskPercent = 0.5;
input double ConfidenceThreshold = 0.65;      // Higher for ensemble
input int MaxTradesPerDay = 5;
input double MaxDailyLoss = 4.0;
input double StopLossUSD = 4.0;
input double TakeProfitUSD = 8.0;
input double MaxMarginPercent = 50.0;

//--- Ensemble Settings
input group "=== Ensemble Configuration ==="
input bool UseEnsemble = true;                // Use both models (requires Transformer)
input double EnsembleAgreementThreshold = 0.60;  // Min confidence when both agree
input double TransformerSignalThreshold = 0.1;   // % change threshold for Transformer
input bool AllowLightGBMFallback = true;      // Use LightGBM alone if Transformer fails

//--- Hybrid Validation Filters
input group "=== Hybrid Validation Filters ==="
input bool EnableHybridValidation = true;
input double MaxSpreadUSD = 2.0;
input double RSI_OverboughtLevel = 70.0;
input double RSI_OversoldLevel = 30.0;
input double ATR_MinLevel = 1.5;
input double ATR_MaxLevel = 8.0;
input double ADX_MinStrength = 20.0;
input bool RequireMTFAlignment = true;

//--- Logging
input group "=== Logging ==="
input bool EnableFeatureLog = false;
input string FeatureLogFile = "feature_log.csv";
input bool EnablePredictionLog = true;
input string PredictionLogFile = "prediction_ensemble_log.csv";
input bool EnableOnnxDebugLogs = false;

CTrade trade;
datetime lastM1BarTime = 0;
int tradesOpenedToday = 0;
datetime lastResetDate = 0;
double startingEquity = 0;

//--- ONNX Models
long g_lightgbm_handle = INVALID_HANDLE;
long g_transformer_handle = INVALID_HANDLE;
bool g_lightgbm_ready = false;
bool g_transformer_ready = false;

//--- Indicator Handles
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

int g_macd_handle = INVALID_HANDLE;
int g_adx_handle = INVALID_HANDLE;

//--- Sequence Buffer for Transformer (30 bars × 130 features)
#define SEQ_LENGTH 30
#define NUM_FEATURES 130
float g_sequence_buffer[SEQ_LENGTH][NUM_FEATURES];
int g_sequence_count = 0;
bool g_sequence_ready = false;

//--- Scaler Parameters for Transformer Normalization
double g_scaler_min[NUM_FEATURES];
double g_scaler_max[NUM_FEATURES];
bool g_scaler_loaded = false;

//+------------------------------------------------------------------+
//| Load ONNX Model with Multiple Path Attempts                      |
//+------------------------------------------------------------------+
long LoadOnnxWithFallback(const string model_file, const uint flags)
{
   const string candidates[] = {
      model_file,
      "Files\\" + model_file,
      "MQL5\\Files\\" + model_file,
      TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + model_file,
      TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + model_file
   };

   PrintFormat("ONNX load: Attempting to load '%s'", model_file);

   for(int i = 0; i < ArraySize(candidates); i++)
   {
      string path = candidates[i];
      ResetLastError();
      long h = OnnxCreate(path, flags);
      if(h != INVALID_HANDLE)
      {
         if(path != model_file)
            PrintFormat("  ✓ Loaded from: '%s'", path);
         return h;
      }
   }

   PrintFormat("  ✗ Failed to load '%s'", model_file);
   return INVALID_HANDLE;
}

//+------------------------------------------------------------------+
//| Load Scaler Parameters from JSON                                 |
//+------------------------------------------------------------------+
bool LoadScalerParams()
{
   // NOTE: MQL5 doesn't have native JSON parsing
   // For now, using hardcoded values or manual parsing
   // TODO: Implement JSON parser or manually create scaler array

   string scaler_file = "transformer_scaler_params.json";
   int handle = FileOpen(scaler_file, FILE_READ|FILE_TXT|FILE_ANSI);

   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: Could not load scaler params from ", scaler_file);
      Print("  Transformer normalization will not be applied");
      Print("  Model performance may be degraded");
      return false;
   }

   // Simple parsing: Look for data_min and data_max arrays
   // This is a simplified version - you may need to enhance for production

   FileClose(handle);

   // For now, initialize with default range [0, 1]
   // You'll need to populate these with actual values from JSON
   for(int i = 0; i < NUM_FEATURES; i++)
   {
      g_scaler_min[i] = 0.0;
      g_scaler_max[i] = 1.0;
   }

   Print("⚠️  Using default scaler params (range [0,1])");
   Print("   For production: Implement proper JSON parsing");

   return true;
}

//+------------------------------------------------------------------+
//| Normalize Features using MinMaxScaler Parameters                 |
//+------------------------------------------------------------------+
void NormalizeFeatures(float &features[], int count)
{
   if(!g_scaler_loaded) return;

   for(int i = 0; i < count && i < NUM_FEATURES; i++)
   {
      double range = g_scaler_max[i] - g_scaler_min[i];
      if(range > 0.0001)  // Avoid division by zero
      {
         features[i] = (float)((features[i] - g_scaler_min[i]) / range);

         // Clip to [0, 1]
         if(features[i] < 0.0) features[i] = 0.0;
         if(features[i] > 1.0) features[i] = 1.0;
      }
   }
}

//+------------------------------------------------------------------+
//| Initialize Indicators                                             |
//+------------------------------------------------------------------+
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

   g_macd_handle = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
   g_adx_handle = iADX(_Symbol, PERIOD_M15, 14);

   if(g_atr14_m1_handle == INVALID_HANDLE || g_rsi14_m1_handle == INVALID_HANDLE ||
      g_ema10_m1_handle == INVALID_HANDLE || g_ema20_m1_handle == INVALID_HANDLE || g_ema50_m1_handle == INVALID_HANDLE ||
      g_ema20_m5_handle == INVALID_HANDLE || g_ema20_m15_handle == INVALID_HANDLE || g_ema20_h1_handle == INVALID_HANDLE ||
      g_ema20_h4_handle == INVALID_HANDLE || g_ema20_d1_handle == INVALID_HANDLE ||
      g_macd_handle == INVALID_HANDLE || g_adx_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return false;
   }

   Print("✓ All indicators initialized");
   return true;
}

//+------------------------------------------------------------------+
//| Initialize ONNX Models                                            |
//+------------------------------------------------------------------+
bool InitModels()
{
   uint flags = EnableOnnxDebugLogs ? ONNX_DEBUG_LOGS : ONNX_DEFAULT;

   // Load LightGBM (required)
   g_lightgbm_handle = LoadOnnxWithFallback("lightgbm_xauusd.onnx", flags);
   if(g_lightgbm_handle == INVALID_HANDLE)
   {
      Print("❌ CRITICAL: LightGBM model failed to load");
      return false;
   }

   // Set LightGBM shapes
   ulong lgb_input_shape[] = {1, 26};
   ulong lgb_label_shape[] = {1};
   ulong lgb_probs_shape[] = {1, 3};

   if(!OnnxSetInputShape(g_lightgbm_handle, 0, lgb_input_shape) ||
      !OnnxSetOutputShape(g_lightgbm_handle, 0, lgb_label_shape) ||
      !OnnxSetOutputShape(g_lightgbm_handle, 1, lgb_probs_shape))
   {
      Print("❌ ERROR: Failed to set LightGBM shapes");
      OnnxRelease(g_lightgbm_handle);
      g_lightgbm_handle = INVALID_HANDLE;
      return false;
   }

   g_lightgbm_ready = true;
   Print("✓ LightGBM model loaded [1,26] → [1] + [1,3]");

   // Load Transformer (optional for ensemble)
   if(UseEnsemble)
   {
      g_transformer_handle = LoadOnnxWithFallback("transformer.onnx", flags);

      if(g_transformer_handle != INVALID_HANDLE)
      {
         // Set Transformer shapes [1, 30, 130] → [1, 1]
         ulong trans_input_shape[] = {1, SEQ_LENGTH, NUM_FEATURES};
         ulong trans_output_shape[] = {1, 1};

         if(!OnnxSetInputShape(g_transformer_handle, 0, trans_input_shape) ||
            !OnnxSetOutputShape(g_transformer_handle, 0, trans_output_shape))
         {
            Print("⚠️  WARNING: Failed to set Transformer shapes");
            OnnxRelease(g_transformer_handle);
            g_transformer_handle = INVALID_HANDLE;
         }
         else
         {
            g_transformer_ready = true;
            Print("✓ Transformer model loaded [1,30,130] → [1,1]");

            // Load scaler parameters
            g_scaler_loaded = LoadScalerParams();
         }
      }
      else
      {
         Print("⚠️  WARNING: Transformer model not found");
         if(AllowLightGBMFallback)
            Print("   Using LightGBM-only mode (fallback enabled)");
         else
         {
            Print("   Ensemble mode required but Transformer unavailable");
            return false;
         }
      }
   }

   return true;
}

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=" * 70);
   Print("XAUUSD NEURAL BOT - ENSEMBLE MODE");
   Print("=" * 70);

   if(!InitIndicators())
   {
      Print("❌ Indicator initialization failed");
      return INIT_FAILED;
   }

   if(!InitModels())
   {
      Print("❌ Model initialization failed");
      return INIT_FAILED;
   }

   startingEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   lastM1BarTime = iTime(_Symbol, PERIOD_M1, 0);

   // Initialize sequence buffer
   ArrayInitialize(g_sequence_buffer, 0.0);
   g_sequence_count = 0;
   g_sequence_ready = false;

   Print("=" * 70);
   Print("✅ ENSEMBLE BOT INITIALIZED");
   Print("=" * 70);
   Print("Mode: ", UseEnsemble && g_transformer_ready ? "ENSEMBLE (LightGBM + Transformer)" : "SINGLE (LightGBM only)");
   Print("Hybrid Validation: ", EnableHybridValidation ? "ENABLED" : "DISABLED");
   Print("Confidence Threshold: ", ConfidenceThreshold * 100, "%");
   PrintFormat("Risk: %.1f%% | SL: $%.2f | TP: $%.2f", RiskPercent, StopLossUSD, TakeProfitUSD);

   if(UseEnsemble && g_transformer_ready)
      Print("⏳ Sequence buffer: Warming up (need ", SEQ_LENGTH, " bars)");

   Print("=" * 70);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_lightgbm_handle != INVALID_HANDLE)
   {
      OnnxRelease(g_lightgbm_handle);
      g_lightgbm_handle = INVALID_HANDLE;
   }

   if(g_transformer_handle != INVALID_HANDLE)
   {
      OnnxRelease(g_transformer_handle);
      g_transformer_handle = INVALID_HANDLE;
   }

   if(g_atr14_m1_handle != INVALID_HANDLE) IndicatorRelease(g_atr14_m1_handle);
   if(g_rsi14_m1_handle != INVALID_HANDLE) IndicatorRelease(g_rsi14_m1_handle);
   if(g_ema10_m1_handle != INVALID_HANDLE) IndicatorRelease(g_ema10_m1_handle);
   if(g_ema20_m1_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_m1_handle);
   if(g_ema50_m1_handle != INVALID_HANDLE) IndicatorRelease(g_ema50_m1_handle);
   if(g_ema20_m5_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_m5_handle);
   if(g_ema20_m15_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_m15_handle);
   if(g_ema20_h1_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_h1_handle);
   if(g_ema20_h4_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_h4_handle);
   if(g_ema20_d1_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_d1_handle);
   if(g_macd_handle != INVALID_HANDLE) IndicatorRelease(g_macd_handle);
   if(g_adx_handle != INVALID_HANDLE) IndicatorRelease(g_adx_handle);

   Print("All resources released");
}

//+------------------------------------------------------------------+
//| Check if new M1 bar                                              |
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
//| Reset daily counters                                             |
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
//| Check risk limits                                                |
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
//| Check if in trading session                                      |
//+------------------------------------------------------------------+
bool IsInSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return (dt.hour >= 12 && dt.hour < 17);
}

//+------------------------------------------------------------------+
//| Calculate 26 features for LightGBM                               |
//+------------------------------------------------------------------+
bool Calculate26Features(float &features[], const int shift)
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
//| Calculate 130 features for Transformer (26 features × 5 TFs)    |
//+------------------------------------------------------------------+
bool Calculate130Features(float &features[], const int shift)
{
   ArrayResize(features, NUM_FEATURES);
   int idx = 0;

   // Calculate 26 features for each of 5 timeframes: M1, M5, M15, H1, H4
   ENUM_TIMEFRAMES timeframes[5] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4};

   for(int tf_idx = 0; tf_idx < 5; tf_idx++)
   {
      ENUM_TIMEFRAMES tf = timeframes[tf_idx];

      // Get OHLC for this timeframe
      double o = iOpen(_Symbol, tf, shift);
      double h = iHigh(_Symbol, tf, shift);
      double l = iLow(_Symbol, tf, shift);
      double c = iClose(_Symbol, tf, shift);

      if(o <= 0 || h <= 0 || l <= 0 || c <= 0) return false;

      // 0-3: Price features
      features[idx++] = (float)(c - o);
      features[idx++] = (float)MathAbs(c - o);
      features[idx++] = (float)(h - l);
      features[idx++] = (float)((c - l) / (h - l + 0.0001));

      // 4-7: Returns
      double c1 = iClose(_Symbol, tf, shift + 1);
      double c5 = iClose(_Symbol, tf, shift + 5);
      double c15 = iClose(_Symbol, tf, shift + 15);
      double c60 = iClose(_Symbol, tf, shift + 60);

      features[idx++] = (float)((c - c1) / (c1 + 0.0001));
      features[idx++] = (float)((c - c5) / (c5 + 0.0001));
      features[idx++] = (float)((c - c15) / (c15 + 0.0001));
      features[idx++] = (float)((c - c60) / (c60 + 0.0001));

      // 8-9: ATR
      double prev_close = iClose(_Symbol, tf, shift + 1);
      double tr = MathMax(h - l, MathMax(MathAbs(h - prev_close), MathAbs(l - prev_close)));
      features[idx++] = (float)tr;

      double atr14_buffer[];
      ArraySetAsSeries(atr14_buffer, true);
      int atr_handle = iATR(_Symbol, tf, 14);
      if(CopyBuffer(atr_handle, 0, shift, 1, atr14_buffer) > 0)
         features[idx++] = (float)atr14_buffer[0];
      else
         features[idx++] = 0.0;
      IndicatorRelease(atr_handle);

      // 10: RSI
      double rsi14_buffer[];
      ArraySetAsSeries(rsi14_buffer, true);
      int rsi_handle = iRSI(_Symbol, tf, 14, PRICE_CLOSE);
      if(CopyBuffer(rsi_handle, 0, shift, 1, rsi14_buffer) > 0)
         features[idx++] = (float)rsi14_buffer[0];
      else
         features[idx++] = 50.0;
      IndicatorRelease(rsi_handle);

      // 11-13: EMAs
      double ema10_buffer[], ema20_buffer[], ema50_buffer[];
      ArraySetAsSeries(ema10_buffer, true);
      ArraySetAsSeries(ema20_buffer, true);
      ArraySetAsSeries(ema50_buffer, true);

      int ema10_h = iMA(_Symbol, tf, 10, 0, MODE_EMA, PRICE_CLOSE);
      int ema20_h = iMA(_Symbol, tf, 20, 0, MODE_EMA, PRICE_CLOSE);
      int ema50_h = iMA(_Symbol, tf, 50, 0, MODE_EMA, PRICE_CLOSE);

      if(CopyBuffer(ema10_h, 0, shift, 1, ema10_buffer) > 0)
         features[idx++] = (float)ema10_buffer[0];
      else
         features[idx++] = (float)c;

      if(CopyBuffer(ema20_h, 0, shift, 1, ema20_buffer) > 0)
         features[idx++] = (float)ema20_buffer[0];
      else
         features[idx++] = (float)c;

      if(CopyBuffer(ema50_h, 0, shift, 1, ema50_buffer) > 0)
         features[idx++] = (float)ema50_buffer[0];
      else
         features[idx++] = (float)c;

      IndicatorRelease(ema10_h);
      IndicatorRelease(ema20_h);
      IndicatorRelease(ema50_h);

      // 14-15: Time features (same for all TFs)
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      features[idx++] = (float)MathSin(2 * M_PI * dt.hour / 24);
      features[idx++] = (float)MathCos(2 * M_PI * dt.hour / 24);

      // 16-25: Multi-timeframe context (relative to current TF)
      // For simplicity, using M5/M15/H1/H4/D1 EMA20 comparison
      double mtf_ema[];
      ArraySetAsSeries(mtf_ema, true);

      int mtf_handles[5] = {
         iMA(_Symbol, PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE),
         iMA(_Symbol, PERIOD_M15, 20, 0, MODE_EMA, PRICE_CLOSE),
         iMA(_Symbol, PERIOD_H1, 20, 0, MODE_EMA, PRICE_CLOSE),
         iMA(_Symbol, PERIOD_H4, 20, 0, MODE_EMA, PRICE_CLOSE),
         iMA(_Symbol, PERIOD_D1, 20, 0, MODE_EMA, PRICE_CLOSE)
      };

      for(int i = 0; i < 5; i++)
      {
         if(CopyBuffer(mtf_handles[i], 0, 1, 1, mtf_ema) > 0)
         {
            double mtf_close = iClose(_Symbol, timeframes[i], 1);
            features[idx++] = (float)((mtf_close > mtf_ema[0]) ? 1.0 : -1.0);

            double mtf_high = iHigh(_Symbol, timeframes[i], 1);
            double mtf_low = iLow(_Symbol, timeframes[i], 1);
            features[idx++] = (float)((mtf_close - mtf_low) / (mtf_high - mtf_low + 0.0001));
         }
         else
         {
            features[idx++] = 0.0;
            features[idx++] = 0.5;
         }

         IndicatorRelease(mtf_handles[i]);
      }
   }

   return (idx == NUM_FEATURES);
}

//+------------------------------------------------------------------+
//| Update sequence buffer with new features                         |
//+------------------------------------------------------------------+
void UpdateSequenceBuffer(float &new_features[])
{
   // Shift buffer: move rows 0-28 to 1-29
   for(int i = SEQ_LENGTH - 1; i > 0; i--)
   {
      for(int j = 0; j < NUM_FEATURES; j++)
      {
         g_sequence_buffer[i][j] = g_sequence_buffer[i-1][j];
      }
   }

   // Insert new features at index 0
   for(int j = 0; j < NUM_FEATURES; j++)
   {
      g_sequence_buffer[0][j] = new_features[j];
   }

   // Update counters
   if(g_sequence_count < SEQ_LENGTH)
      g_sequence_count++;

   if(g_sequence_count >= SEQ_LENGTH)
      g_sequence_ready = true;
}

//+------------------------------------------------------------------+
//| Predict using LightGBM (26 features)                            |
//+------------------------------------------------------------------+
int PredictLightGBM(double &confidence)
{
   confidence = 0.0;

   if(!g_lightgbm_ready || g_lightgbm_handle == INVALID_HANDLE)
      return 1; // HOLD

   // Calculate 26 features
   float features[];
   if(!Calculate26Features(features, 1))
      return 1;

   // Run inference
   static float input_data[26];
   static long output_label[1];
   static float output_probs[3];

   ArrayCopy(input_data, features, 0, 0, 26);

   if(!OnnxRun(g_lightgbm_handle, ONNX_DEFAULT, input_data, output_label, output_probs))
   {
      Print("ERROR: LightGBM inference failed");
      return 1;
   }

   // Extract probabilities
   double p_short = output_probs[0];
   double p_hold = output_probs[1];
   double p_long = output_probs[2];

   // Validate probabilities
   double prob_sum = p_short + p_hold + p_long;
   if(prob_sum < 0.99 || prob_sum > 1.01)
   {
      Print("WARNING: LightGBM invalid probabilities, sum=", prob_sum);
      return 1;
   }

   // Find best prediction
   int best = 0;
   double best_p = p_short;

   if(p_hold > best_p) { best_p = p_hold; best = 1; }
   if(p_long > best_p) { best_p = p_long; best = 2; }

   confidence = best_p;
   return best;
}

//+------------------------------------------------------------------+
//| Predict using Transformer (30×130 sequence)                     |
//+------------------------------------------------------------------+
double PredictTransformer()
{
   if(!g_transformer_ready || g_transformer_handle == INVALID_HANDLE || !g_sequence_ready)
      return 0.0; // Neutral (no prediction)

   // Prepare sequence input [1, 30, 130]
   static float input_seq[SEQ_LENGTH * NUM_FEATURES];

   // Flatten sequence buffer
   int idx = 0;
   for(int i = 0; i < SEQ_LENGTH; i++)
   {
      for(int j = 0; j < NUM_FEATURES; j++)
      {
         input_seq[idx++] = g_sequence_buffer[i][j];
      }
   }

   // Apply normalization if scaler loaded
   if(g_scaler_loaded)
   {
      idx = 0;
      for(int i = 0; i < SEQ_LENGTH; i++)
      {
         for(int j = 0; j < NUM_FEATURES; j++)
         {
            double range = g_scaler_max[j] - g_scaler_min[j];
            if(range > 0.0001)
            {
               float val = input_seq[idx];
               val = (float)((val - g_scaler_min[j]) / range);
               if(val < 0.0) val = 0.0;
               if(val > 1.0) val = 1.0;
               input_seq[idx] = val;
            }
            idx++;
         }
      }
   }

   // Run inference (output: price change prediction)
   static float output_pred[1];

   if(!OnnxRun(g_transformer_handle, ONNX_DEFAULT, input_seq, output_pred))
   {
      Print("ERROR: Transformer inference failed");
      return 0.0;
   }

   return output_pred[0]; // Return predicted price change
}

//+------------------------------------------------------------------+
//| Ensemble prediction: LightGBM + Transformer voting              |
//+------------------------------------------------------------------+
int PredictEnsemble(double &confidence)
{
   confidence = 0.0;

   // 1. Get LightGBM prediction
   double lgb_conf = 0.0;
   int lgb_signal = PredictLightGBM(lgb_conf);

   // 2. If ensemble disabled or Transformer not ready, use LightGBM only
   if(!UseEnsemble || !g_transformer_ready || !g_sequence_ready)
   {
      confidence = lgb_conf;
      return lgb_signal;
   }

   // 3. Get Transformer prediction (price change)
   double trans_pred = PredictTransformer();

   // 4. Convert Transformer prediction to signal
   int trans_signal = 1; // HOLD
   if(trans_pred > TransformerSignalThreshold)
      trans_signal = 2; // LONG
   else if(trans_pred < -TransformerSignalThreshold)
      trans_signal = 0; // SHORT

   // 5. Ensemble voting: Both must agree
   if(lgb_signal == trans_signal)
   {
      // Agreement: Use LightGBM confidence (higher for ensemble)
      if(lgb_conf >= EnsembleAgreementThreshold)
      {
         confidence = lgb_conf;
         Print("✓ ENSEMBLE AGREEMENT: ", lgb_signal == 2 ? "LONG" : (lgb_signal == 0 ? "SHORT" : "HOLD"),
               " (LGB: ", lgb_conf, ", Trans: ", trans_pred, ")");
         return lgb_signal;
      }
   }
   else
   {
      Print("ENSEMBLE DISAGREE: LGB=", lgb_signal, " Trans=", trans_signal, " → HOLD");
   }

   // No agreement or low confidence: HOLD
   return 1;
}

//+------------------------------------------------------------------+
//| Calculate position lot size                                     |
//+------------------------------------------------------------------+
double CalculateLotSize(ENUM_ORDER_TYPE orderType)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskAmount = equity * RiskPercent / 100.0;

   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double stopLossPoints = StopLossUSD;

   double riskPerLot = (stopLossPoints / tickSize) * tickValue;

   if(riskPerLot <= 0)
      return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   double lotSize = riskAmount / riskPerLot;

   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lotSize = MathFloor(lotSize / lotStep) * lotStep;

   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(lotSize < minLot) lotSize = minLot;
   if(lotSize > maxLot) lotSize = maxLot;

   // Margin check
   double margin_required = 0;
   double price = (orderType == ORDER_TYPE_BUY) ?
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                  SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(!OrderCalcMargin(orderType, _Symbol, lotSize, price, margin_required))
      return minLot;

   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double maxMarginRatio = MaxMarginPercent / 100.0;

   if(margin_required > freeMargin * maxMarginRatio)
   {
      double scale = (freeMargin * maxMarginRatio) / margin_required;
      lotSize = MathFloor((lotSize * scale) / lotStep) * lotStep;
      if(lotSize < minLot) lotSize = 0;
   }

   return lotSize;
}

//+------------------------------------------------------------------+
//| Hybrid Validation: LONG signals                                 |
//+------------------------------------------------------------------+
bool ValidateLongSignal(double ml_confidence)
{
   if(!EnableHybridValidation)
      return true;

   // 1. Spread filter
   double spread_usd = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                        SymbolInfoDouble(_Symbol, SYMBOL_BID));

   if(spread_usd > MaxSpreadUSD)
   {
      Print("FILTER REJECT [LONG]: Spread=$", DoubleToString(spread_usd, 2));
      return false;
   }

   // 2. RSI filter
   double rsi[];
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(g_rsi14_m1_handle, 0, 0, 1, rsi) > 0)
   {
      if(rsi[0] > RSI_OverboughtLevel)
      {
         Print("FILTER REJECT [LONG]: RSI overbought=", rsi[0]);
         return false;
      }
   }

   // 3. MACD alignment
   double macd_main[], macd_signal[];
   ArraySetAsSeries(macd_main, true);
   ArraySetAsSeries(macd_signal, true);

   if(CopyBuffer(g_macd_handle, 0, 0, 1, macd_main) > 0 &&
      CopyBuffer(g_macd_handle, 1, 0, 1, macd_signal) > 0)
   {
      if(macd_main[0] < macd_signal[0])
      {
         Print("FILTER REJECT [LONG]: MACD bearish");
         return false;
      }
   }

   // 4. ADX trend strength
   double adx[];
   ArraySetAsSeries(adx, true);
   if(CopyBuffer(g_adx_handle, 0, 0, 1, adx) > 0)
   {
      if(adx[0] < ADX_MinStrength)
      {
         Print("FILTER REJECT [LONG]: ADX weak=", adx[0]);
         return false;
      }
   }

   // 5. ATR volatility filter
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(g_atr14_m1_handle, 0, 0, 1, atr) > 0)
   {
      if(atr[0] < ATR_MinLevel || atr[0] > ATR_MaxLevel)
      {
         Print("FILTER REJECT [LONG]: ATR=", atr[0]);
         return false;
      }
   }

   // 6. Multi-timeframe EMA alignment
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
            Print("FILTER REJECT [LONG]: Price below MTF EMAs");
            return false;
         }
      }
   }

   Print("✓ LONG signal validated (conf=", ml_confidence, ")");
   return true;
}

//+------------------------------------------------------------------+
//| Hybrid Validation: SHORT signals                                |
//+------------------------------------------------------------------+
bool ValidateShortSignal(double ml_confidence)
{
   if(!EnableHybridValidation)
      return true;

   // 1. Spread filter
   double spread_usd = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                        SymbolInfoDouble(_Symbol, SYMBOL_BID));

   if(spread_usd > MaxSpreadUSD)
   {
      Print("FILTER REJECT [SHORT]: Spread=$", DoubleToString(spread_usd, 2));
      return false;
   }

   // 2. RSI filter
   double rsi[];
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(g_rsi14_m1_handle, 0, 0, 1, rsi) > 0)
   {
      if(rsi[0] < RSI_OversoldLevel)
      {
         Print("FILTER REJECT [SHORT]: RSI oversold=", rsi[0]);
         return false;
      }
   }

   // 3. MACD alignment
   double macd_main[], macd_signal[];
   ArraySetAsSeries(macd_main, true);
   ArraySetAsSeries(macd_signal, true);

   if(CopyBuffer(g_macd_handle, 0, 0, 1, macd_main) > 0 &&
      CopyBuffer(g_macd_handle, 1, 0, 1, macd_signal) > 0)
   {
      if(macd_main[0] > macd_signal[0])
      {
         Print("FILTER REJECT [SHORT]: MACD bullish");
         return false;
      }
   }

   // 4. ADX trend strength
   double adx[];
   ArraySetAsSeries(adx, true);
   if(CopyBuffer(g_adx_handle, 0, 0, 1, adx) > 0)
   {
      if(adx[0] < ADX_MinStrength)
      {
         Print("FILTER REJECT [SHORT]: ADX weak=", adx[0]);
         return false;
      }
   }

   // 5. ATR volatility filter
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(g_atr14_m1_handle, 0, 0, 1, atr) > 0)
   {
      if(atr[0] < ATR_MinLevel || atr[0] > ATR_MaxLevel)
      {
         Print("FILTER REJECT [SHORT]: ATR=", atr[0]);
         return false;
      }
   }

   // 6. Multi-timeframe EMA alignment
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
            Print("FILTER REJECT [SHORT]: Price above MTF EMAs");
            return false;
         }
      }
   }

   Print("✓ SHORT signal validated (conf=", ml_confidence, ")");
   return true;
}

//+------------------------------------------------------------------+
//| OnTick - Main trading logic                                     |
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

   // Update Transformer sequence buffer (if using ensemble)
   if(UseEnsemble && g_transformer_ready)
   {
      float features130[];
      if(Calculate130Features(features130, 1))
      {
         UpdateSequenceBuffer(features130);
      }
   }

   // Get ensemble prediction
   double confidence;
   int signal = PredictEnsemble(confidence);

   // Manage existing position
   if(PositionSelect(_Symbol))
   {
      // TODO: Implement trailing stop
      return;
   }

   // Open new position
   if(signal == 2) // LONG
   {
      if(!ValidateLongSignal(confidence))
      {
         Print("ML LONG rejected by hybrid validation");
         return;
      }

      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - StopLossUSD;
      double tp = price + TakeProfitUSD;
      double lots = CalculateLotSize(ORDER_TYPE_BUY);

      if(lots <= 0)
      {
         Print("SKIP LONG: Insufficient margin");
         return;
      }

      if(trade.Buy(lots, _Symbol, price, sl, tp,
                   StringFormat("ENSEMBLE LONG %.2f", confidence)))
      {
         tradesOpenedToday++;
         PrintFormat("✅ ENSEMBLE LONG: %.2f lots @ %.5f (conf: %.2f)",
                     lots, price, confidence);
      }
   }
   else if(signal == 0) // SHORT
   {
      if(!ValidateShortSignal(confidence))
      {
         Print("ML SHORT rejected by hybrid validation");
         return;
      }

      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + StopLossUSD;
      double tp = price - TakeProfitUSD;
      double lots = CalculateLotSize(ORDER_TYPE_SELL);

      if(lots <= 0)
      {
         Print("SKIP SHORT: Insufficient margin");
         return;
      }

      if(trade.Sell(lots, _Symbol, price, sl, tp,
                    StringFormat("ENSEMBLE SHORT %.2f", confidence)))
      {
         tradesOpenedToday++;
         PrintFormat("✅ ENSEMBLE SHORT: %.2f lots @ %.5f (conf: %.2f)",
                     lots, price, confidence);
      }
   }
}
//+------------------------------------------------------------------+
