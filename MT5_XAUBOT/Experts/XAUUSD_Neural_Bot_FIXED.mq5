//+------------------------------------------------------------------+
//|                           XAUUSD_Neural_Bot_FIXED.mq5            |
//|                    PROPER ONNX Implementation (MQL5 Best Practices)|
//+------------------------------------------------------------------+
#property copyright "XAUBOT - Fixed ONNX Integration"
#property version   "3.00"
#property description "Neural Network with proper ONNX setup"

//--- Input Parameters
input double   InpRiskPercent = 0.5;
input double   InpConfidenceThreshold = 0.35;
input int      InpMaxTradesPerDay = 10;
input double   InpATRMultiplierSL = 1.5;
input double   InpRiskRewardRatio = 2.0;
input int      InpMagicNumber = 230172;
input string   InpModelPath = "lightgbm_xauusd.onnx";

//--- Global Variables
long           g_onnx_handle = INVALID_HANDLE;
int            g_atr_handle = INVALID_HANDLE;
int            g_rsi_handle = INVALID_HANDLE;
int            g_ema10_handle = INVALID_HANDLE;
int            g_ema20_handle = INVALID_HANDLE;
int            g_ema50_handle = INVALID_HANDLE;

datetime       g_last_bar_time = 0;
int            g_daily_trades = 0;
datetime       g_current_day = 0;

//--- ONNX Constants (CRITICAL!)
#define INPUT_SIZE  26
#define OUTPUT_SIZE 3
#define BATCH_SIZE  1

//+------------------------------------------------------------------+
//| Expert initialization - PROPER ONNX SETUP                        |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("XAUUSD Neural Bot v3.0 (FIXED ONNX)");
    Print("Following MQL5 Best Practices");
    Print("========================================");

    //--- Step 1: Load ONNX model with multiple fallback paths
    string model_paths[] = {
        InpModelPath,
        "Files\\" + InpModelPath,
        "MQL5\\Files\\" + InpModelPath,
        TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + InpModelPath,
        TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + InpModelPath
    };

    bool model_loaded = false;
    for(int i = 0; i < ArraySize(model_paths); i++)
    {
        ResetLastError();
        g_onnx_handle = OnnxCreate(model_paths[i], ONNX_DEFAULT);

        if(g_onnx_handle != INVALID_HANDLE)
        {
            Print("✓ Model loaded from: ", model_paths[i]);
            model_loaded = true;
            break;
        }
        else
        {
            int error = GetLastError();
            Print("  Path ", i+1, " failed (", error, "): ", model_paths[i]);
        }
    }

    if(!model_loaded)
    {
        Print("ERROR: Failed to load ONNX model from all paths!");
        Print("Searched paths:");
        for(int i = 0; i < ArraySize(model_paths); i++)
            Print("  ", i+1, ". ", model_paths[i]);
        return INIT_FAILED;
    }

    //--- Step 2: CRITICAL - Set explicit input/output shapes
    //    This is required per MQL5 article!
    Print("Setting ONNX input/output shapes...");

    if(!OnnxSetInputShape(g_onnx_handle, 0, BATCH_SIZE, INPUT_SIZE))
    {
        Print("ERROR: Failed to set input shape [", BATCH_SIZE, ", ", INPUT_SIZE, "]");
        Print("Error code: ", GetLastError());
        return INIT_FAILED;
    }

    if(!OnnxSetOutputShape(g_onnx_handle, 0, BATCH_SIZE, OUTPUT_SIZE))
    {
        Print("ERROR: Failed to set output shape [", BATCH_SIZE, ", ", OUTPUT_SIZE, "]");
        Print("Error code: ", GetLastError());
        return INIT_FAILED;
    }

    Print("✓ ONNX shapes configured: Input[", BATCH_SIZE, ",", INPUT_SIZE, "] → Output[", BATCH_SIZE, ",", OUTPUT_SIZE, "]");

    //--- Step 3: Test ONNX model with dummy data
    Print("Testing ONNX inference with dummy data...");
    float test_input[INPUT_SIZE];
    float test_output[OUTPUT_SIZE];

    // Fill with test values
    for(int i = 0; i < INPUT_SIZE; i++)
        test_input[i] = (float)MathRandomUniform(0.0, 1.0);

    // Resize output array (REQUIRED per article!)
    ArrayResize(test_output, OUTPUT_SIZE);

    if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, test_input, test_output))
    {
        Print("ERROR: ONNX test inference failed!");
        Print("Error code: ", GetLastError());
        return INIT_FAILED;
    }

    // Verify output
    float prob_sum = 0.0;
    Print("Test prediction output:");
    for(int i = 0; i < OUTPUT_SIZE; i++)
    {
        Print("  Class ", i, ": ", test_output[i]);
        prob_sum += test_output[i];
    }
    Print("  Sum: ", prob_sum, " (should be ~1.0 for probabilities)");

    if(prob_sum < 0.95 || prob_sum > 1.05)
    {
        Print("WARNING: Output doesn't look like probabilities!");
        Print("Expected sum ~1.0, got: ", prob_sum);
    }
    else
    {
        Print("✓ ONNX model test passed!");
    }

    //--- Step 4: Initialize indicators
    g_atr_handle = iATR(_Symbol, PERIOD_M1, 14);
    g_rsi_handle = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
    g_ema10_handle = iMA(_Symbol, PERIOD_M1, 10, 0, MODE_EMA, PRICE_CLOSE);
    g_ema20_handle = iMA(_Symbol, PERIOD_M1, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_ema50_handle = iMA(_Symbol, PERIOD_M1, 50, 0, MODE_EMA, PRICE_CLOSE);

    if(g_atr_handle == INVALID_HANDLE || g_rsi_handle == INVALID_HANDLE ||
       g_ema10_handle == INVALID_HANDLE || g_ema20_handle == INVALID_HANDLE ||
       g_ema50_handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to initialize indicators");
        return INIT_FAILED;
    }

    Print("✓ All indicators initialized");

    //--- Configuration summary
    Print("========================================");
    Print("Configuration:");
    Print("  Model: ", InpModelPath);
    Print("  Input shape: [", BATCH_SIZE, ", ", INPUT_SIZE, "]");
    Print("  Output shape: [", BATCH_SIZE, ", ", OUTPUT_SIZE, "]");
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
    if(g_onnx_handle != INVALID_HANDLE)
        OnnxRelease(g_onnx_handle);

    if(g_atr_handle != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
    if(g_rsi_handle != INVALID_HANDLE) IndicatorRelease(g_rsi_handle);
    if(g_ema10_handle != INVALID_HANDLE) IndicatorRelease(g_ema10_handle);
    if(g_ema20_handle != INVALID_HANDLE) IndicatorRelease(g_ema20_handle);
    if(g_ema50_handle != INVALID_HANDLE) IndicatorRelease(g_ema50_handle);

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
        return;

    g_last_bar_time = current_bar_time;

    //--- Reset daily counter
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

    //--- Position/trade checks
    if(PositionsTotal() > 0) return;
    if(g_daily_trades >= InpMaxTradesPerDay) return;

    //--- Get prediction
    float features[INPUT_SIZE];
    if(!CalculateFeatures(features))
        return;

    int signal;
    double confidence;
    if(!GetMLPrediction(features, signal, confidence))
        return;

    //--- Filter trades
    if(confidence < InpConfidenceThreshold) return;
    if(signal == 1) return;  // Skip HOLD

    //--- Open trade
    OpenTrade(signal, confidence);
}

//+------------------------------------------------------------------+
//| Calculate 26 features (FLOAT array for ONNX)                    |
//+------------------------------------------------------------------+
bool CalculateFeatures(float &features[])
{
    ArrayResize(features, INPUT_SIZE);  // Explicit resize
    ArrayInitialize(features, 0.0);

    //--- Get price data
    MqlRates rates[];
    if(CopyRates(_Symbol, PERIOD_M1, 1, 100, rates) < 100)
        return false;

    int idx = 99;

    //--- Get indicators
    double atr[], rsi[], ema10[], ema20[], ema50[];
    if(CopyBuffer(g_atr_handle, 0, 1, 100, atr) < 100) return false;
    if(CopyBuffer(g_rsi_handle, 0, 1, 100, rsi) < 100) return false;
    if(CopyBuffer(g_ema10_handle, 0, 1, 100, ema10) < 100) return false;
    if(CopyBuffer(g_ema20_handle, 0, 1, 100, ema20) < 100) return false;
    if(CopyBuffer(g_ema50_handle, 0, 1, 100, ema50) < 100) return false;

    //--- Calculate features (convert to float)
    double body = rates[idx].close - rates[idx].open;
    features[0] = (float)body;
    features[1] = (float)MathAbs(body);
    features[2] = (float)(rates[idx].high - rates[idx].low);
    features[3] = (float)((rates[idx].close - rates[idx].low) / (rates[idx].high - rates[idx].low + 1e-8));

    //--- Returns
    features[4] = (float)((rates[idx].close / rates[idx-1].close) - 1.0);
    features[5] = (float)((rates[idx].close / rates[idx-5].close) - 1.0);
    features[6] = (float)((rates[idx].close / rates[idx-15].close) - 1.0);
    features[7] = (float)((rates[idx].close / rates[idx-60].close) - 1.0);

    //--- TR
    double tr = MathMax(rates[idx].high - rates[idx].low,
                MathMax(MathAbs(rates[idx].high - rates[idx-1].close),
                        MathAbs(rates[idx].low - rates[idx-1].close)));

    //--- Indicators
    features[8] = (float)tr;
    features[9] = (float)atr[idx];
    features[10] = (float)rsi[idx];
    features[11] = (float)ema10[idx];
    features[12] = (float)ema20[idx];
    features[13] = (float)ema50[idx];

    //--- Time
    TimeToStruct(rates[idx].time, dt);
    features[14] = (float)MathSin(2.0 * M_PI * dt.hour / 24.0);
    features[15] = (float)MathCos(2.0 * M_PI * dt.hour / 24.0);

    //--- Placeholders (16-25)
    for(int i = 16; i < INPUT_SIZE; i++)
        features[i] = 0.0;

    return true;
}

//+------------------------------------------------------------------+
//| Get ML prediction with PROPER error handling                    |
//+------------------------------------------------------------------+
bool GetMLPrediction(const float &features[], int &signal, double &confidence)
{
    //--- Prepare output array (MUST resize first!)
    float output_probs[OUTPUT_SIZE];
    ArrayResize(output_probs, OUTPUT_SIZE);
    ArrayInitialize(output_probs, 0.0);

    //--- Run ONNX inference
    if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, features, output_probs))
    {
        int error = GetLastError();
        Print("ERROR: ONNX prediction failed! Error code: ", error);
        return false;
    }

    //--- Verify output
    if(ArraySize(output_probs) != OUTPUT_SIZE)
    {
        Print("ERROR: Invalid output size: ", ArraySize(output_probs), " expected: ", OUTPUT_SIZE);
        return false;
    }

    //--- Find class with max probability
    signal = 0;
    confidence = output_probs[0];

    for(int i = 1; i < OUTPUT_SIZE; i++)
    {
        if(output_probs[i] > confidence)
        {
            signal = i;
            confidence = output_probs[i];
        }
    }

    //--- Debug output (first prediction only)
    static bool first_prediction = true;
    if(first_prediction)
    {
        Print("First prediction:");
        Print("  Output[0] (HOLD): ", output_probs[0]);
        Print("  Output[1] (LONG): ", output_probs[1]);
        Print("  Output[2] (SHORT): ", output_probs[2]);
        Print("  Signal: ", signal, " Confidence: ", confidence);
        first_prediction = false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Open trade                                                       |
//+------------------------------------------------------------------+
void OpenTrade(int signal, double confidence)
{
    // ... (same as before)
    Print("Opening trade: Signal=", signal, " Confidence=", confidence);
}
//+------------------------------------------------------------------+
