//+------------------------------------------------------------------+
//|                                    XAUUSD_NeuralBot_DLL.mq5      |
//|                        Using LightGBM DLL instead of ONNX        |
//+------------------------------------------------------------------+
#property copyright "XAUBOT - LightGBM DLL"
#property version   "4.01"
#property description "LightGBM via DLL - INVERTED PREDICTIONS"

#include <Trade\Trade.mqh>

//--- Input Parameters
input double   InpRiskPercent = 1.0;
input double   InpConfidenceThreshold = 0.55;
input int      InpMaxTradesPerDay = 10;
input double   InpATRMultiplierSL = 1.5;
input double   InpRiskRewardRatio = 3.0;
input int      InpMagicNumber = 230172;
input string   InpModelPath = "lightgbm_xauusd.txt";  // Text model file

//--- DLL Imports
#import "lightgbm_mt5.dll"
   int LoadModel(string model_path);
   int Predict(double &features[], double &probs[]);
   int UnloadModel();
   int GetModelInfo(string &info);
#import

//--- Global Variables
bool           g_model_loaded = false;
int            g_atr_handle = INVALID_HANDLE;
int            g_rsi_handle = INVALID_HANDLE;
int            g_ema10_handle = INVALID_HANDLE;
int            g_ema20_handle = INVALID_HANDLE;
int            g_ema50_handle = INVALID_HANDLE;

// Multi-timeframe EMA handles
int            g_ema20_m5_handle = INVALID_HANDLE;
int            g_ema20_m15_handle = INVALID_HANDLE;
int            g_ema20_h1_handle = INVALID_HANDLE;
int            g_ema20_h4_handle = INVALID_HANDLE;
int            g_ema20_d1_handle = INVALID_HANDLE;

datetime       g_last_bar_time = 0;
int            g_daily_trades = 0;
datetime       g_current_day = 0;

CTrade         trade;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("XAUUSD Neural Bot v4.01 (INVERTED)");
    Print("LightGBM DLL - Predictions Inverted");
    Print("========================================");

    //--- Load LightGBM model via DLL
    string model_full_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH) +
                             "\\Files\\" + InpModelPath;

    Print("Loading model from: ", model_full_path);

    int result = LoadModel(model_full_path);

    if(result == 1)
    {
        g_model_loaded = true;
        Print("✓ Model loaded successfully via DLL");

        // Get model info
        string info = "";
        GetModelInfo(info);
        Print("  ", info);
    }
    else
    {
        Print("ERROR: Failed to load model via DLL");
        Print("  Make sure lightgbm_mt5.dll is in MQL5\\Libraries\\");
        Print("  And model file is in Terminal\\Common\\Files\\");
        return INIT_FAILED;
    }

    //--- Initialize indicators
    g_atr_handle = iATR(_Symbol, PERIOD_M1, 14);
    g_rsi_handle = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
    g_ema10_handle = iMA(_Symbol, PERIOD_M1, 10, 0, MODE_EMA, PRICE_CLOSE);
    g_ema20_handle = iMA(_Symbol, PERIOD_M1, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_ema50_handle = iMA(_Symbol, PERIOD_M1, 50, 0, MODE_EMA, PRICE_CLOSE);

    g_ema20_m5_handle  = iMA(_Symbol, PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_ema20_m15_handle = iMA(_Symbol, PERIOD_M15, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_ema20_h1_handle  = iMA(_Symbol, PERIOD_H1, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_ema20_h4_handle  = iMA(_Symbol, PERIOD_H4, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_ema20_d1_handle  = iMA(_Symbol, PERIOD_D1, 20, 0, MODE_EMA, PRICE_CLOSE);

    if(g_atr_handle == INVALID_HANDLE || g_rsi_handle == INVALID_HANDLE ||
       g_ema10_handle == INVALID_HANDLE || g_ema20_handle == INVALID_HANDLE ||
       g_ema50_handle == INVALID_HANDLE ||
       g_ema20_m5_handle == INVALID_HANDLE || g_ema20_m15_handle == INVALID_HANDLE ||
       g_ema20_h1_handle == INVALID_HANDLE || g_ema20_h4_handle == INVALID_HANDLE ||
       g_ema20_d1_handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to initialize indicators");
        return INIT_FAILED;
    }

    Print("✓ All indicators initialized");

    //--- Configuration summary
    Print("========================================");
    Print("Configuration:");
    Print("  Model: ", InpModelPath);
    Print("  Risk: ", InpRiskPercent, "%");
    Print("  Confidence: ", InpConfidenceThreshold);
    Print("  ATR SL: ", InpATRMultiplierSL);
    Print("  RR Ratio: ", InpRiskRewardRatio, ":1");
    Print("========================================");
    Print("Bot initialized successfully!");
    Print("========================================");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(g_model_loaded)
    {
        UnloadModel();
        Print("Model unloaded");
    }

    if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
    if(g_rsi_handle != INVALID_HANDLE) IndicatorRelease(g_rsi_handle);
    if(g_ema10_handle != INVALID_HANDLE) IndicatorRelease(g_ema10_handle);
    if(g_ema20_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_handle);
    if(g_ema50_handle != INVALID_HANDLE) IndicatorRelease(g_ema50_handle);
    if(g_ema20_m5_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_m5_handle);
    if(g_ema20_m15_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_m15_handle);
    if(g_ema20_h1_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_h1_handle);
    if(g_ema20_h4_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_h4_handle);
    if(g_ema20_d1_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_d1_handle);

    Print("XAUUSD Neural Bot stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Calculate 26 features                                            |
//+------------------------------------------------------------------+
bool Calculate26Features(double &features[])
{
    ArrayResize(features, 26);

    // Get M1 OHLC
    double o = iOpen(_Symbol, PERIOD_M1, 1);
    double h = iHigh(_Symbol, PERIOD_M1, 1);
    double l = iLow(_Symbol, PERIOD_M1, 1);
    double c = iClose(_Symbol, PERIOD_M1, 1);

    if(o <= 0 || h <= 0 || l <= 0 || c <= 0) return false;

    int idx = 0;

    // 0-3: Price features
    features[idx++] = c - o;
    features[idx++] = MathAbs(c - o);
    features[idx++] = h - l;
    features[idx++] = (c - l) / (h - l + 0.0001);

    // 4-7: Returns
    features[idx++] = (c - iClose(_Symbol, PERIOD_M1, 2)) / (iClose(_Symbol, PERIOD_M1, 2) + 0.0001);
    features[idx++] = (c - iClose(_Symbol, PERIOD_M1, 6)) / (iClose(_Symbol, PERIOD_M1, 6) + 0.0001);
    features[idx++] = (c - iClose(_Symbol, PERIOD_M1, 16)) / (iClose(_Symbol, PERIOD_M1, 16) + 0.0001);
    features[idx++] = (c - iClose(_Symbol, PERIOD_M1, 61)) / (iClose(_Symbol, PERIOD_M1, 61) + 0.0001);

    // 8-9: ATR
    double prev_close = iClose(_Symbol, PERIOD_M1, 2);
    double tr = MathMax(h - l, MathMax(MathAbs(h - prev_close), MathAbs(l - prev_close)));
    features[idx++] = tr;

    double atr[];
    ArraySetAsSeries(atr, true);
    if(CopyBuffer(g_atr_handle, 0, 1, 1, atr) <= 0) return false;
    features[idx++] = atr[0];

    // 10: RSI
    double rsi[];
    ArraySetAsSeries(rsi, true);
    if(CopyBuffer(g_rsi_handle, 0, 1, 1, rsi) <= 0) return false;
    features[idx++] = rsi[0];

    // 11-13: EMAs
    double ema10[], ema20[], ema50[];
    ArraySetAsSeries(ema10, true);
    ArraySetAsSeries(ema20, true);
    ArraySetAsSeries(ema50, true);

    if(CopyBuffer(g_ema10_handle, 0, 1, 1, ema10) <= 0) return false;
    if(CopyBuffer(g_ema20_handle, 0, 1, 1, ema20) <= 0) return false;
    if(CopyBuffer(g_ema50_handle, 0, 1, 1, ema50) <= 0) return false;

    features[idx++] = ema10[0];
    features[idx++] = ema20[0];
    features[idx++] = ema50[0];

    // 14-15: Time features
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    features[idx++] = MathSin(2 * M_PI * dt.hour / 24);
    features[idx++] = MathCos(2 * M_PI * dt.hour / 24);

    // 16-25: Multi-timeframe features
    // M5, M15, H1, H4, D1 (trend and position for each)
    double ema20_values[];
    ArraySetAsSeries(ema20_values, true);

    int tf_handles[] = {g_ema20_m5_handle, g_ema20_m15_handle, g_ema20_h1_handle,
                        g_ema20_h4_handle, g_ema20_d1_handle};
    ENUM_TIMEFRAMES tfs[] = {PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1};

    for(int i = 0; i < 5; i++)
    {
        if(CopyBuffer(tf_handles[i], 0, 1, 1, ema20_values) <= 0) return false;

        double tf_close = iClose(_Symbol, tfs[i], 1);
        double tf_high = iHigh(_Symbol, tfs[i], 1);
        double tf_low = iLow(_Symbol, tfs[i], 1);

        features[idx++] = (tf_close > ema20_values[0]) ? 1.0 : -1.0;  // Trend
        features[idx++] = (tf_close - tf_low) / (tf_high - tf_low + 0.0001);  // Position
    }

    return true;
}

//+------------------------------------------------------------------+
//| Get prediction from LightGBM                                     |
//+------------------------------------------------------------------+
int PredictSignal(double &out_confidence)
{
    out_confidence = 0.0;

    if(!g_model_loaded)
        return 1; // HOLD

    // Calculate features
    double features[];
    if(!Calculate26Features(features))
    {
        Print("ERROR: Failed to calculate features");
        return 1;
    }

    // Get prediction
    double probs[3];  // [SHORT, HOLD, LONG]

    if(Predict(features, probs) != 1)
    {
        Print("ERROR: DLL prediction failed");
        return 1;
    }

    // Find best class
    int best_class = 0;
    double best_prob = probs[0];

    for(int i = 1; i < 3; i++)
    {
        if(probs[i] > best_prob)
        {
            best_prob = probs[i];
            best_class = i;
        }
    }

    // Apply confidence threshold
    if(best_prob < InpConfidenceThreshold)
        return 1; // HOLD

    out_confidence = best_prob;

    // INVERTED PREDICTIONS - Testing if model is predicting opposite
    // If model says SHORT (0), trade LONG (2)
    // If model says LONG (2), trade SHORT (0)
    // If model says HOLD (1), keep HOLD (1)
    if(best_class == 0)
        return 2; // Model said SHORT, we go LONG
    else if(best_class == 2)
        return 0; // Model said LONG, we go SHORT
    else
        return 1; // Model said HOLD, we HOLD
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check for new bar
    datetime current_bar = iTime(_Symbol, PERIOD_M1, 0);
    if(current_bar == g_last_bar_time)
        return;

    g_last_bar_time = current_bar;

    // Reset daily counter
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    datetime current_day = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

    if(current_day != g_current_day)
    {
        g_daily_trades = 0;
        g_current_day = current_day;
    }

    // Check daily limit
    if(g_daily_trades >= InpMaxTradesPerDay)
        return;

    // Check if position exists
    if(PositionSelect(_Symbol))
        return;  // Already in position

    // Get prediction
    double confidence;
    int signal = PredictSignal(confidence);

    // Execute trades based on signal
    double atr[];
    ArraySetAsSeries(atr, true);
    CopyBuffer(g_atr_handle, 0, 0, 1, atr);

    double price_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double price_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    if(signal == 2)  // LONG
    {
        double sl = price_ask - (InpATRMultiplierSL * atr[0]);
        double tp = price_ask + (InpRiskRewardRatio * InpATRMultiplierSL * atr[0]);

        if(trade.Buy(0.01, _Symbol, price_ask, sl, tp, StringFormat("LONG %.2f", confidence)))
        {
            g_daily_trades++;
            PrintFormat("LONG: %.2f @ %.5f (SL:%.5f TP:%.5f) Conf:%.2f",
                        0.01, price_ask, sl, tp, confidence);
        }
    }
    else if(signal == 0)  // SHORT
    {
        double sl = price_bid + (InpATRMultiplierSL * atr[0]);
        double tp = price_bid - (InpRiskRewardRatio * InpATRMultiplierSL * atr[0]);

        if(trade.Sell(0.01, _Symbol, price_bid, sl, tp, StringFormat("SHORT %.2f", confidence)))
        {
            g_daily_trades++;
            PrintFormat("SHORT: %.2f @ %.5f (SL:%.5f TP:%.5f) Conf:%.2f",
                        0.01, price_bid, sl, tp, confidence);
        }
    }
}
//+------------------------------------------------------------------+
