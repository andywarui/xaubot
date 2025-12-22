//+------------------------------------------------------------------+
//|                                    XAUUSD_NeuralBot_Hybrid.mq5   |
//|        Phase 3: Hybrid Transformer + LightGBM Architecture       |
//|        2-Model Pipeline: Transformer -> multi_tf_signal -> LGB   |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Neural Bot Hybrid v2.0"
#property version   "2.00"
#property description "Hybrid architecture using Transformer for multi-TF signal generation"
#property description "and LightGBM for final classification (BUY/SELL/HOLD)"
#property strict

#include <Trade\Trade.mqh>
#include "Include\FeatureCalculator.mqh"
#include "Include\SequenceBuffer.mqh"

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Trading Parameters ==="
input double RiskPercent = 0.5;              // Risk per trade (%)
input double ConfidenceThreshold = 0.60;     // Min prediction confidence
input int    MaxTradesPerDay = 5;            // Max trades per day
input double MaxDailyLoss = 4.0;             // Max daily loss (%)
input double StopLossUSD = 4.0;              // Stop loss in USD
input double TakeProfitUSD = 8.0;            // Take profit in USD
input double MaxMarginPercent = 50.0;        // Max margin per trade (%)

input group "=== Model Settings ==="
input string TransformerModel = "NeuralBot\\transformer.onnx";     // Transformer ONNX path
input string LightGBMModel = "NeuralBot\\hybrid_lightgbm.onnx";    // LightGBM ONNX path
input string ScalerFile = "NeuralBot\\scaler_params.json";         // Scaler JSON path
input int    SequenceLength = 30;            // Transformer sequence length
input int    TransformerFeatures = 130;      // Features per timestep

input group "=== Session & Logging ==="
input bool   UseSessionFilter = false;       // Only trade during session (12-17 UTC)
input bool   EnableFeatureLog = false;       // Log features to file
input bool   EnablePredictionLog = false;    // Log predictions to file
input bool   EnableDebugLogs = false;        // Verbose debug output
input string LogDirectory = "NeuralBot\\logs";  // Log file directory

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade trade;

// Models and components
CTransformerInference g_transformer;
CLightGBMInference    g_lightgbm;
CFeatureCalculator    g_features;
CMinMaxScaler         g_scaler;
CSequenceBuffer       g_sequence;

// State tracking
datetime g_lastM1BarTime = 0;
int      g_tradesOpenedToday = 0;
datetime g_lastResetDate = 0;
double   g_startingEquity = 0;
bool     g_modelReady = false;
int      g_warmupBarsRemaining = 0;

// Feature log handles
int g_featureLogHandle = INVALID_HANDLE;
int g_predictionLogHandle = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=============================================================");
   Print("XAUUSD Neural Bot Hybrid v2.0 - Initializing...");
   Print("=============================================================");
   
   // 1. Initialize feature calculator
   Print("1. Initializing feature calculator...");
   if(!g_features.Init(_Symbol))
   {
      Print("ERROR: Failed to initialize feature calculator");
      return(INIT_FAILED);
   }
   
   // 2. Load scaler
   Print("2. Loading scaler from: ", ScalerFile);
   if(!g_scaler.Load(ScalerFile))
   {
      Print("WARNING: Failed to load scaler - using raw features");
      // Continue without scaler (may affect accuracy)
   }
   else
   {
      PrintFormat("   Scaler loaded: %d features", g_scaler.GetNFeatures());
   }
   
   // 3. Initialize sequence buffer
   Print("3. Initializing sequence buffer...");
   if(!g_sequence.Init(SequenceLength, TransformerFeatures))
   {
      Print("ERROR: Failed to initialize sequence buffer");
      return(INIT_FAILED);
   }
   
   // 4. Load Transformer model
   Print("4. Loading Transformer: ", TransformerModel);
   if(!g_transformer.Load(TransformerModel, SequenceLength, TransformerFeatures, EnableDebugLogs))
   {
      Print("ERROR: Failed to load Transformer model");
      Print("   Make sure file exists in MQL5/Files/", TransformerModel);
      return(INIT_FAILED);
   }
   
   // 5. Load LightGBM model
   Print("5. Loading LightGBM: ", LightGBMModel);
   if(!g_lightgbm.Load(LightGBMModel, 27, EnableDebugLogs))  // 27 features (with multi_tf_signal)
   {
      Print("ERROR: Failed to load LightGBM model");
      Print("   Make sure file exists in MQL5/Files/", LightGBMModel);
      return(INIT_FAILED);
   }
   
   // 6. Initialize logging if enabled
   if(EnableFeatureLog)
      InitFeatureLog();
   if(EnablePredictionLog)
      InitPredictionLog();
   
   // 7. Warm up sequence buffer with historical data
   Print("6. Warming up sequence buffer with ", SequenceLength, " bars...");
   g_warmupBarsRemaining = SequenceLength;
   WarmUpSequenceBuffer();
   
   // Initialize state
   g_startingEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_lastM1BarTime = iTime(_Symbol, PERIOD_M1, 0);
   g_modelReady = true;
   
   Print("=============================================================");
   Print("INITIALIZATION COMPLETE");
   Print("=============================================================");
   PrintFormat("Transformer: [1, %d, %d] -> [1, 1]", SequenceLength, TransformerFeatures);
   PrintFormat("LightGBM: [1, 27] -> [label, probs]");
   PrintFormat("Confidence threshold: %.0f%%", ConfidenceThreshold * 100);
   PrintFormat("Risk: %.1f%% | SL: $%.2f | TP: $%.2f", RiskPercent, StopLossUSD, TakeProfitUSD);
   Print("Ready to trade. Waiting for M1 bars...");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_transformer.Release();
   g_lightgbm.Release();
   g_features.Deinit();
   
   if(g_featureLogHandle != INVALID_HANDLE)
   {
      FileClose(g_featureLogHandle);
      g_featureLogHandle = INVALID_HANDLE;
   }
   if(g_predictionLogHandle != INVALID_HANDLE)
   {
      FileClose(g_predictionLogHandle);
      g_predictionLogHandle = INVALID_HANDLE;
   }
   
   Print("XAUUSD Neural Bot Hybrid - Deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!g_modelReady) return;
   
   // Check for new M1 bar
   if(!IsNewM1Bar()) return;
   
   // Reset daily counters at midnight
   ResetDailyCounters();
   
   // Update sequence buffer with new bar
   if(!UpdateSequenceBuffer())
   {
      if(EnableDebugLogs)
         Print("DEBUG: Failed to update sequence buffer");
      return;
   }
   
   // Check if buffer is ready
   if(!g_sequence.IsFull())
   {
      if(EnableDebugLogs)
         PrintFormat("DEBUG: Warming up - %d/%d bars", g_sequence.GetCurrentCount(), SequenceLength);
      return;
   }
   
   // Session filter
   if(UseSessionFilter && !IsInSession())
      return;
   
   // Risk limits
   if(!CheckRiskLimits())
      return;
   
   // Run 2-model prediction
   double confidence = 0;
   int signal = PredictSignal(confidence);
   
   // Log prediction
   if(EnablePredictionLog)
      LogPrediction(signal, confidence);
   
   // Execute trade if signal meets threshold
   if(signal != 0 && confidence >= ConfidenceThreshold)  // 0 = HOLD
   {
      ExecuteTrade(signal, confidence);
   }
}

//+------------------------------------------------------------------+
//| Warm up sequence buffer with historical data                      |
//+------------------------------------------------------------------+
void WarmUpSequenceBuffer()
{
   for(int i = SequenceLength; i >= 1; i--)
   {
      float raw_features[];
      if(!g_features.CalculateTransformerFeatures(raw_features, i))
      {
         if(EnableDebugLogs)
            PrintFormat("DEBUG: WarmUp - Failed to calc features at shift %d", i);
         continue;
      }
      
      float scaled_features[];
      if(g_scaler.IsLoaded())
         g_scaler.Transform(raw_features, scaled_features);
      else
         ArrayCopy(scaled_features, raw_features);
      
      g_sequence.Push(scaled_features);
   }
   
   if(g_sequence.IsFull())
      Print("   Sequence buffer warmed up successfully");
   else
      PrintFormat("   Sequence buffer partially filled: %d/%d", g_sequence.GetCurrentCount(), SequenceLength);
}

//+------------------------------------------------------------------+
//| Update sequence buffer with new bar                               |
//+------------------------------------------------------------------+
bool UpdateSequenceBuffer()
{
   // Calculate features for the last closed bar (shift=1)
   float raw_features[];
   if(!g_features.CalculateTransformerFeatures(raw_features, 1))
      return false;
   
   // Scale features
   float scaled_features[];
   if(g_scaler.IsLoaded())
   {
      if(!g_scaler.Transform(raw_features, scaled_features))
         return false;
   }
   else
   {
      ArrayCopy(scaled_features, raw_features);
   }
   
   // Log features if enabled
   if(EnableFeatureLog)
      LogFeatures(raw_features, scaled_features);
   
   // Push to sequence buffer
   return g_sequence.Push(scaled_features);
}

//+------------------------------------------------------------------+
//| Run 2-model prediction pipeline                                   |
//+------------------------------------------------------------------+
int PredictSignal(double &out_confidence)
{
   out_confidence = 0;
   
   // Step 1: Get sequence from buffer
   float sequence[];
   if(!g_sequence.GetOnnxInput(sequence))
   {
      if(EnableDebugLogs)
         Print("DEBUG: Failed to get sequence input");
      return 0;  // HOLD
   }
   
   // Step 2: Run Transformer to get multi_tf_signal
   float multi_tf_signal = 0;
   if(!g_transformer.Predict(sequence, multi_tf_signal))
   {
      Print("ERROR: Transformer inference failed");
      return 0;  // HOLD
   }
   
   if(EnableDebugLogs)
      PrintFormat("DEBUG: Transformer multi_tf_signal = %.6f", multi_tf_signal);
   
   // Step 3: Calculate 26 LightGBM features (without multi_tf_signal)
   float lgb_features_26[];
   if(!g_features.CalculateLightGBMFeatures(lgb_features_26, 1))
   {
      Print("ERROR: Failed to calculate LightGBM features");
      return 0;  // HOLD
   }
   
   // Step 4: Combine into 27 features (multi_tf_signal at index 0)
   float lgb_input[27];
   lgb_input[0] = multi_tf_signal;
   for(int i = 0; i < 26; i++)
      lgb_input[i + 1] = lgb_features_26[i];
   
   // Step 5: Run LightGBM inference
   int label = 0;
   float probs[];
   if(!g_lightgbm.Predict(lgb_input, label, probs))
   {
      Print("ERROR: LightGBM inference failed");
      return 0;  // HOLD
   }
   
   // Validate probabilities
   double prob_sum = probs[0] + probs[1] + probs[2];
   if(prob_sum < 0.99 || prob_sum > 1.01)
   {
      PrintFormat("WARNING: Invalid probability sum %.4f", prob_sum);
      return 0;  // HOLD
   }
   
   // Find max probability
   double max_prob = probs[0];
   int best_class = 0;
   for(int i = 1; i < 3; i++)
   {
      if(probs[i] > max_prob)
      {
         max_prob = probs[i];
         best_class = i;
      }
   }
   
   out_confidence = max_prob;
   
   if(EnableDebugLogs)
   {
      PrintFormat("DEBUG: LightGBM output - label=%d, probs=[%.4f, %.4f, %.4f], best=%d (%.2f%%)",
                  label, probs[0], probs[1], probs[2], best_class, max_prob * 100);
   }
   
   // Return signal: 0=HOLD, 1=BUY, 2=SELL
   return best_class;
}

//+------------------------------------------------------------------+
//| Execute trade based on signal                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(const int signal, const double confidence)
{
   // Check existing positions
   if(PositionsTotal() > 0)
   {
      if(EnableDebugLogs)
         Print("DEBUG: Position already open, skipping");
      return;
   }
   
   // Calculate lot size
   double lots = CalculateLotSize(signal == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   if(lots <= 0)
   {
      Print("WARNING: Invalid lot size calculated");
      return;
   }
   
   // Calculate SL/TP
   double price = 0;
   double sl = 0;
   double tp = 0;
   
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   // Convert USD to points
   double slPoints = StopLossUSD / tickSize;
   double tpPoints = TakeProfitUSD / tickSize;
   
   if(signal == 1)  // BUY
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - slPoints * point;
      tp = price + tpPoints * point;
      
      if(trade.Buy(lots, _Symbol, price, sl, tp, StringFormat("NeuralBot BUY %.0f%%", confidence * 100)))
      {
         g_tradesOpenedToday++;
         PrintFormat("BUY executed: %.2f lots @ %.5f, SL=%.5f, TP=%.5f, Conf=%.1f%%", 
                     lots, price, sl, tp, confidence * 100);
      }
   }
   else if(signal == 2)  // SELL
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + slPoints * point;
      tp = price - tpPoints * point;
      
      if(trade.Sell(lots, _Symbol, price, sl, tp, StringFormat("NeuralBot SELL %.0f%%", confidence * 100)))
      {
         g_tradesOpenedToday++;
         PrintFormat("SELL executed: %.2f lots @ %.5f, SL=%.5f, TP=%.5f, Conf=%.1f%%", 
                     lots, price, sl, tp, confidence * 100);
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                  |
//+------------------------------------------------------------------+
double CalculateLotSize(const ENUM_ORDER_TYPE orderType)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskAmount = equity * RiskPercent / 100.0;
   
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double stopLossPoints = StopLossUSD / tickSize;
   
   if(tickValue <= 0 || stopLossPoints <= 0)
      return 0;
   
   double riskPerLot = stopLossPoints * tickValue;
   double lots = riskAmount / riskPerLot;
   
   // Apply min/max constraints
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lots = MathFloor(lots / stepLot) * stepLot;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   
   // Check margin constraint
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double maxLotsMargin = freeMargin * MaxMarginPercent / 100.0;
   
   double marginRequired = 0;
   if(!OrderCalcMargin(orderType, _Symbol, lots, SymbolInfoDouble(_Symbol, SYMBOL_ASK), marginRequired))
      return 0;
   
   if(marginRequired > maxLotsMargin)
   {
      lots = lots * maxLotsMargin / marginRequired;
      lots = MathFloor(lots / stepLot) * stepLot;
      lots = MathMax(minLot, lots);
   }
   
   return lots;
}

//+------------------------------------------------------------------+
//| Check if new M1 bar                                               |
//+------------------------------------------------------------------+
bool IsNewM1Bar()
{
   datetime currentTime = iTime(_Symbol, PERIOD_M1, 0);
   if(currentTime != g_lastM1BarTime)
   {
      g_lastM1BarTime = currentTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Reset daily counters                                              |
//+------------------------------------------------------------------+
void ResetDailyCounters()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime currentDate = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
   
   if(currentDate != g_lastResetDate)
   {
      g_tradesOpenedToday = 0;
      g_startingEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      g_lastResetDate = currentDate;
   }
}

//+------------------------------------------------------------------+
//| Check risk limits                                                 |
//+------------------------------------------------------------------+
bool CheckRiskLimits()
{
   if(g_tradesOpenedToday >= MaxTradesPerDay)
   {
      if(EnableDebugLogs)
         Print("DEBUG: Max trades per day reached");
      return false;
   }
   
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyLoss = (g_startingEquity - currentEquity) / g_startingEquity * 100;
   
   if(dailyLoss >= MaxDailyLoss)
   {
      if(EnableDebugLogs)
         PrintFormat("DEBUG: Max daily loss reached (%.1f%%)", dailyLoss);
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if in trading session                                       |
//+------------------------------------------------------------------+
bool IsInSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return (dt.hour >= 12 && dt.hour < 17);
}

//+------------------------------------------------------------------+
//| Initialize feature log                                            |
//+------------------------------------------------------------------+
void InitFeatureLog()
{
   string filename = LogDirectory + "\\features_" + TimeToString(TimeCurrent(), TIME_DATE) + ".csv";
   StringReplace(filename, ".", "");
   StringReplace(filename, ":", "");
   
   g_featureLogHandle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ';');
   if(g_featureLogHandle != INVALID_HANDLE)
   {
      // Write header
      string header = "time";
      for(int i = 0; i < TransformerFeatures; i++)
         header += ";raw_" + IntegerToString(i);
      for(int i = 0; i < TransformerFeatures; i++)
         header += ";scaled_" + IntegerToString(i);
      FileWrite(g_featureLogHandle, header);
   }
}

//+------------------------------------------------------------------+
//| Log features                                                      |
//+------------------------------------------------------------------+
void LogFeatures(const float &raw[], const float &scaled[])
{
   if(g_featureLogHandle == INVALID_HANDLE) return;
   
   string line = TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES | TIME_SECONDS);
   
   for(int i = 0; i < ArraySize(raw); i++)
      line += ";" + DoubleToString(raw[i], 6);
   for(int i = 0; i < ArraySize(scaled); i++)
      line += ";" + DoubleToString(scaled[i], 6);
   
   FileWrite(g_featureLogHandle, line);
   FileFlush(g_featureLogHandle);
}

//+------------------------------------------------------------------+
//| Initialize prediction log                                         |
//+------------------------------------------------------------------+
void InitPredictionLog()
{
   string filename = LogDirectory + "\\predictions_" + TimeToString(TimeCurrent(), TIME_DATE) + ".csv";
   StringReplace(filename, ".", "");
   StringReplace(filename, ":", "");
   
   g_predictionLogHandle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ';');
   if(g_predictionLogHandle != INVALID_HANDLE)
   {
      FileWrite(g_predictionLogHandle, "time;signal;confidence;action");
   }
}

//+------------------------------------------------------------------+
//| Log prediction                                                    |
//+------------------------------------------------------------------+
void LogPrediction(const int signal, const double confidence)
{
   if(g_predictionLogHandle == INVALID_HANDLE) return;
   
   string signalStr = (signal == 0) ? "HOLD" : (signal == 1) ? "BUY" : "SELL";
   string actionStr = (confidence >= ConfidenceThreshold && signal != 0) ? "TRADE" : "SKIP";
   
   FileWrite(g_predictionLogHandle, 
             TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES | TIME_SECONDS),
             signalStr,
             DoubleToString(confidence, 4),
             actionStr);
   FileFlush(g_predictionLogHandle);
}
