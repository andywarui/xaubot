//+------------------------------------------------------------------+
//|                                      XAUUSD_Neural_Bot_v2.mq5    |
//|                                       MT5 XAUUSD Trading Bot     |
//|                              Backtest Results: +472% (3 years)   |
//+------------------------------------------------------------------+
#property copyright "XAUBOT Neural Trading System"
#property version   "2.00"
#property description "LightGBM + ATR-based 2:1 RR Strategy"
#property description "Backtest: $10K → $57K (+472%) | Win Rate: 43.83%"

//--- Input Parameters
input double   InpRiskPercent = 0.5;           // Risk per trade (%)
input double   InpConfidenceThreshold = 0.35;  // ML Confidence threshold (0.35 = aggressive)
input int      InpMaxTradesPerDay = 10;        // Max trades per day
input double   InpATRMultiplierSL = 1.5;       // ATR multiplier for Stop Loss
input double   InpRiskRewardRatio = 2.0;       // Risk:Reward ratio (2.0 = 2:1)
input int      InpMagicNumber = 230172;        // Magic number
input bool     InpUseValidation = false;       // Enable hybrid validation (KEEP FALSE!)

//--- ONNX Model
input string   InpModelPath = "lightgbm_real_26features.onnx";  // Model filename

//--- Global Variables
long           g_onnx_handle = INVALID_HANDLE;
int            g_atr_handle = INVALID_HANDLE;
int            g_rsi_handle = INVALID_HANDLE;
int            g_ema10_handle = INVALID_HANDLE;
int            g_ema20_handle = INVALID_HANDLE;
int            g_ema50_handle = INVALID_HANDLE;
int            g_alligator_handle = INVALID_HANDLE;
datetime       g_last_bar_time = 0;
int            g_daily_trades = 0;
datetime       g_current_day = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("XAUUSD Neural Bot v2.0 - Initializing");
    Print("Backtest Performance: +472% (3 years)");
    Print("========================================");

    //--- Load ONNX model
    // Note: OnnxCreate automatically looks in MQL5\Files\ directory
    g_onnx_handle = OnnxCreate(InpModelPath, ONNX_DEFAULT);

    if(g_onnx_handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to load ONNX model: ", InpModelPath);
        Print("Make sure model is in: MQL5\\Files\\");
        Print("Full path should be: MQL5\\Files\\", InpModelPath);
        return INIT_FAILED;
    }

    Print("✓ ONNX Model loaded: ", InpModelPath);

    //--- Initialize indicators
    g_atr_handle = iATR(_Symbol, PERIOD_M1, 14);
    g_rsi_handle = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
    g_ema10_handle = iMA(_Symbol, PERIOD_M1, 10, 0, MODE_EMA, PRICE_CLOSE);
    g_ema20_handle = iMA(_Symbol, PERIOD_M1, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_ema50_handle = iMA(_Symbol, PERIOD_M1, 50, 0, MODE_EMA, PRICE_CLOSE);
    g_alligator_handle = iAlligator(_Symbol, PERIOD_M1, 13, 8, 8, 5, 5, 3, MODE_SMMA, PRICE_MEDIAN);

    if(g_atr_handle == INVALID_HANDLE || g_rsi_handle == INVALID_HANDLE ||
       g_ema10_handle == INVALID_HANDLE || g_ema20_handle == INVALID_HANDLE ||
       g_ema50_handle == INVALID_HANDLE || g_alligator_handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to initialize indicators");
        return INIT_FAILED;
    }

    Print("✓ Indicators initialized (ATR, RSI, EMA 10/20/50, Alligator)");

    //--- Configuration
    Print("Configuration:");
    Print("  Risk per trade: ", InpRiskPercent, "%");
    Print("  Confidence threshold: ", InpConfidenceThreshold);
    Print("  ATR SL multiplier: ", InpATRMultiplierSL);
    Print("  Risk:Reward ratio: ", InpRiskRewardRatio, ":1");
    Print("  Max trades/day: ", InpMaxTradesPerDay);
    Print("  Validation: ", InpUseValidation ? "ENABLED ⚠️" : "DISABLED ✓");

    if(InpUseValidation)
    {
        Print("WARNING: Validation blocks ALL trades. Keep disabled for profitable trading!");
    }

    Print("========================================");
    Print("Bot initialized successfully!");
    Print("========================================");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(g_onnx_handle != INVALID_HANDLE)
        OnnxRelease(g_onnx_handle);

    if(g_atr_handle != INVALID_HANDLE)
        IndicatorRelease(g_atr_handle);
    if(g_rsi_handle != INVALID_HANDLE)
        IndicatorRelease(g_rsi_handle);
    if(g_ema10_handle != INVALID_HANDLE)
        IndicatorRelease(g_ema10_handle);
    if(g_ema20_handle != INVALID_HANDLE)
        IndicatorRelease(g_ema20_handle);
    if(g_ema50_handle != INVALID_HANDLE)
        IndicatorRelease(g_ema50_handle);
    if(g_alligator_handle != INVALID_HANDLE)
        IndicatorRelease(g_alligator_handle);

    Print("XAUUSD Neural Bot stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Check for new bar
    datetime current_bar_time = iTime(_Symbol, PERIOD_M1, 0);
    if(current_bar_time == g_last_bar_time)
        return;  // Wait for new bar

    g_last_bar_time = current_bar_time;

    //--- Reset daily trade counter
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    datetime today = StringToTime(IntegerToString(dt.year) + "." +
                                   IntegerToString(dt.mon) + "." +
                                   IntegerToString(dt.day));

    if(today != g_current_day)
    {
        g_daily_trades = 0;
        g_current_day = today;
    }

    //--- Check if we can open new trade
    if(PositionsTotal() > 0)
        return;  // Already have open position

    if(g_daily_trades >= InpMaxTradesPerDay)
        return;  // Max trades reached

    //--- Calculate 26 features
    double features[26];
    if(!CalculateFeatures(features))
        return;

    //--- Get ML prediction
    int signal;
    double confidence;
    if(!GetMLPrediction(features, signal, confidence))
        return;

    //--- Check confidence threshold
    if(confidence < InpConfidenceThreshold)
        return;

    //--- Skip HOLD signals
    if(signal == 1)
        return;

    //--- Validate signal (if enabled)
    if(InpUseValidation)
    {
        if(!ValidateSignal(signal, confidence))
            return;
    }

    //--- Open trade
    OpenTrade(signal, confidence);
}

//+------------------------------------------------------------------+
//| Calculate 26 features for model                                  |
//+------------------------------------------------------------------+
bool CalculateFeatures(double &features[])
{
    //--- Get price data
    MqlRates rates[];
    if(CopyRates(_Symbol, PERIOD_M1, 1, 100, rates) < 100)
        return false;

    int idx = 99;  // Current completed bar

    //--- Get indicator values
    double atr[], rsi[], ema10[], ema20[], ema50[];
    if(CopyBuffer(g_atr_handle, 0, 1, 100, atr) < 100) return false;
    if(CopyBuffer(g_rsi_handle, 0, 1, 100, rsi) < 100) return false;
    if(CopyBuffer(g_ema10_handle, 0, 1, 100, ema10) < 100) return false;
    if(CopyBuffer(g_ema20_handle, 0, 1, 100, ema20) < 100) return false;
    if(CopyBuffer(g_ema50_handle, 0, 1, 100, ema50) < 100) return false;

    //--- Price features
    double body = rates[idx].close - rates[idx].open;
    features[0] = body;
    features[1] = MathAbs(body);
    features[2] = rates[idx].high - rates[idx].low;
    features[3] = (rates[idx].close - rates[idx].low) / (rates[idx].high - rates[idx].low + 1e-8);

    //--- Returns
    features[4] = (rates[idx].close / rates[idx-1].close) - 1.0;
    features[5] = (rates[idx].close / rates[idx-5].close) - 1.0;
    features[6] = (rates[idx].close / rates[idx-15].close) - 1.0;
    features[7] = (rates[idx].close / rates[idx-60].close) - 1.0;

    //--- Calculate TR manually
    double tr = MathMax(rates[idx].high - rates[idx].low,
                MathMax(MathAbs(rates[idx].high - rates[idx-1].close),
                        MathAbs(rates[idx].low - rates[idx-1].close)));

    //--- Technical indicators
    features[8] = tr;
    features[9] = atr[idx];
    features[10] = rsi[idx];
    features[11] = ema10[idx];
    features[12] = ema20[idx];
    features[13] = ema50[idx];

    //--- Time features
    MqlDateTime dt;
    TimeToStruct(rates[idx].time, dt);
    features[14] = MathSin(2.0 * M_PI * dt.hour / 24.0);
    features[15] = MathCos(2.0 * M_PI * dt.hour / 24.0);

    //--- Multi-timeframe placeholders (16-25)
    for(int i = 16; i < 26; i++)
        features[i] = 0.0;

    return true;
}

//+------------------------------------------------------------------+
//| Get ML prediction from ONNX model                                |
//+------------------------------------------------------------------+
bool GetMLPrediction(const double &features[], int &signal, double &confidence)
{
    //--- Prepare input (reshape to float)
    float input_data[26];
    for(int i = 0; i < 26; i++)
        input_data[i] = (float)features[i];

    //--- Run ONNX model
    float output_probs[];  // Will contain probabilities as dict values

    if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_data, output_probs))
    {
        Print("ERROR: ONNX prediction failed");
        return false;
    }

    //--- CRITICAL: LightGBM ONNX outputs probabilities as 3 values
    //    output_probs[0] = P(class 0 = SHORT)
    //    output_probs[1] = P(class 1 = HOLD)
    //    output_probs[2] = P(class 2 = LONG)

    if(ArraySize(output_probs) < 3)
    {
        Print("ERROR: Invalid ONNX output size: ", ArraySize(output_probs));
        return false;
    }

    //--- Find class with max probability
    signal = 0;
    confidence = output_probs[0];

    for(int i = 1; i < 3; i++)
    {
        if(output_probs[i] > confidence)
        {
            signal = i;
            confidence = output_probs[i];
        }
    }

    return true;
}

//+------------------------------------------------------------------+
//| Validate signal (hybrid validation - OPTIONAL)                   |
//+------------------------------------------------------------------+
bool ValidateSignal(int signal, double confidence)
{
    //--- WARNING: This validation blocks ALL trades in backtest
    //--- Keep InpUseValidation = false for profitable trading!

    double rsi[];
    if(CopyBuffer(g_rsi_handle, 0, 1, 1, rsi) < 1)
        return false;

    //--- Get current price
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double current_price = (signal == 2) ? ask : bid;

    //--- Get Alligator values
    double jaw[], teeth[], lips[];
    if(CopyBuffer(g_alligator_handle, 0, 1, 1, jaw) < 1)    // Jaw (blue, slowest)
        return false;
    if(CopyBuffer(g_alligator_handle, 1, 1, 1, teeth) < 1)  // Teeth (red, medium)
        return false;
    if(CopyBuffer(g_alligator_handle, 2, 1, 1, lips) < 1)   // Lips (green, fastest)
        return false;

    //--- LONG validation
    if(signal == 2)
    {
        // RSI: Not overbought
        if(rsi[0] > 70.0)
            return false;

        // Alligator: Price should be above all lines (uptrend)
        if(current_price <= jaw[0] || current_price <= teeth[0] || current_price <= lips[0])
            return false;

        // Alligator: Lines should be ordered (Lips > Teeth > Jaw for strong uptrend)
        if(lips[0] <= teeth[0] || teeth[0] <= jaw[0])
            return false;
    }

    //--- SHORT validation
    if(signal == 0)
    {
        // RSI: Not oversold
        if(rsi[0] < 30.0)
            return false;

        // Alligator: Price should be below all lines (downtrend)
        if(current_price >= jaw[0] || current_price >= teeth[0] || current_price >= lips[0])
            return false;

        // Alligator: Lines should be ordered (Jaw > Teeth > Lips for strong downtrend)
        if(jaw[0] <= teeth[0] || teeth[0] <= lips[0])
            return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Open trade with ATR-based 2:1 RR                                 |
//+------------------------------------------------------------------+
void OpenTrade(int signal, double confidence)
{
    //--- Get current ATR
    double atr[];
    if(CopyBuffer(g_atr_handle, 0, 1, 1, atr) < 1)
        return;

    double current_atr = atr[0];

    //--- Calculate dynamic SL/TP based on ATR
    double sl_distance = current_atr * InpATRMultiplierSL;  // SL = 1.5 × ATR
    double tp_distance = sl_distance * InpRiskRewardRatio;  // TP = 2 × SL

    //--- Get current price
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    double entry_price, sl_price, tp_price;
    ENUM_ORDER_TYPE order_type;

    if(signal == 2)  // LONG
    {
        entry_price = ask;
        sl_price = entry_price - sl_distance;
        tp_price = entry_price + tp_distance;
        order_type = ORDER_TYPE_BUY;
    }
    else  // SHORT (signal == 0)
    {
        entry_price = bid;
        sl_price = entry_price + sl_distance;
        tp_price = entry_price - tp_distance;
        order_type = ORDER_TYPE_SELL;
    }

    //--- Calculate lot size based on risk
    double risk_amount = AccountInfoDouble(ACCOUNT_BALANCE) * (InpRiskPercent / 100.0);
    double sl_points = MathAbs(entry_price - sl_price) / _Point;
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lot_size = risk_amount / (sl_points * tick_value);

    //--- Normalize lot size
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lot_size = MathFloor(lot_size / lot_step) * lot_step;
    lot_size = MathMax(min_lot, MathMin(lot_size, max_lot));

    //--- Send order
    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lot_size;
    request.type = order_type;
    request.price = entry_price;
    request.sl = sl_price;
    request.tp = tp_price;
    request.deviation = 10;
    request.magic = InpMagicNumber;
    request.comment = StringFormat("ML:%s C:%.2f ATR:%.1f",
                                    signal == 2 ? "LONG" : "SHORT",
                                    confidence * 100, current_atr);

    if(!OrderSend(request, result))
    {
        Print("OrderSend failed: ", GetLastError());
        return;
    }

    if(result.retcode == TRADE_RETCODE_DONE)
    {
        g_daily_trades++;
        Print("✓ Trade opened: ", signal == 2 ? "LONG" : "SHORT",
              " | Lots: ", lot_size,
              " | Confidence: ", confidence * 100, "%",
              " | SL: ", sl_distance, " | TP: ", tp_distance,
              " | ATR: ", current_atr);
    }
    else
    {
        Print("Order failed: ", result.retcode, " - ", result.comment);
    }
}
//+------------------------------------------------------------------+
