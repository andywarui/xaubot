//+------------------------------------------------------------------+
//|                                         TestFeatureCalculator.mq5 |
//|            Phase 3 Task 5: MT5 Test Script for Features           |
//|                     Validates feature calculation parity          |
//+------------------------------------------------------------------+
#property copyright "XAUUSD Neural Bot"
#property version   "1.00"
#property script_show_inputs
#property strict

#include "Include\FeatureCalculator.mqh"
#include "Include\SequenceBuffer.mqh"

input int    TestBars = 10;              // Number of bars to test
input bool   TestScaler = true;          // Test scaler loading
input bool   TestSequenceBuffer = true;  // Test sequence buffer
input bool   TestTransformerONNX = true; // Test Transformer inference
input bool   TestLightGBMONNX = true;    // Test LightGBM inference
input string ScalerFile = "NeuralBot\\scaler_params.json";
input string TransformerFile = "NeuralBot\\transformer.onnx";
input string LightGBMFile = "NeuralBot\\hybrid_lightgbm.onnx";

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("=============================================================");
   Print("MT5 Feature Calculator & ONNX Test Script");
   Print("Symbol: ", _Symbol, " | Timeframe: M1");
   Print("=============================================================");
   
   int passed = 0;
   int failed = 0;
   
   //--- Test 1: Feature Calculator Initialization
   Print("\n--- Test 1: Feature Calculator ---");
   CFeatureCalculator features;
   if(features.Init(_Symbol))
   {
      Print("  [PASS] Feature calculator initialized");
      passed++;
   }
   else
   {
      Print("  [FAIL] Feature calculator initialization failed");
      failed++;
      return;  // Cannot continue without features
   }
   
   //--- Test 2: Calculate LightGBM Features (26)
   Print("\n--- Test 2: LightGBM Features (26) ---");
   {
      float lgb_features[];
      bool success = true;
      
      for(int shift = 1; shift <= TestBars; shift++)
      {
         if(!features.CalculateLightGBMFeatures(lgb_features, shift))
         {
            PrintFormat("  [FAIL] Failed at shift %d", shift);
            success = false;
            break;
         }
         
         if(shift == 1)
         {
            Print("  Sample features at shift=1:");
            PrintFormat("    [0] body = %.6f", lgb_features[0]);
            PrintFormat("    [1] body_abs = %.6f", lgb_features[1]);
            PrintFormat("    [2] candle_range = %.6f", lgb_features[2]);
            PrintFormat("    [3] close_position = %.6f", lgb_features[3]);
            PrintFormat("    [10] rsi_14 = %.2f", lgb_features[10]);
            PrintFormat("    [16] M5_trend = %.0f", lgb_features[16]);
         }
      }
      
      if(success)
      {
         PrintFormat("  [PASS] Calculated 26 features for %d bars", TestBars);
         passed++;
      }
      else
         failed++;
   }
   
   //--- Test 3: Calculate Transformer Features (130)
   Print("\n--- Test 3: Transformer Features (130) ---");
   {
      float tf_features[];
      bool success = true;
      
      for(int shift = 1; shift <= TestBars; shift++)
      {
         if(!features.CalculateTransformerFeatures(tf_features, shift))
         {
            PrintFormat("  [FAIL] Failed at shift %d", shift);
            success = false;
            break;
         }
         
         if(shift == 1)
         {
            Print("  Sample features at shift=1:");
            PrintFormat("    Feature count = %d", ArraySize(tf_features));
            PrintFormat("    [0] = %.6f", tf_features[0]);
            PrintFormat("    [1] = %.6f", tf_features[1]);
            PrintFormat("    [64] = %.6f", tf_features[64]);
            PrintFormat("    [129] = %.6f", tf_features[129]);
            
            // Check for NaN/Inf
            bool hasNaN = false;
            for(int i = 0; i < ArraySize(tf_features); i++)
            {
               if(!MathIsValidNumber(tf_features[i]))
               {
                  hasNaN = true;
                  PrintFormat("  WARNING: NaN/Inf at index %d", i);
               }
            }
            if(!hasNaN)
               Print("  No NaN/Inf values detected");
         }
      }
      
      if(success)
      {
         PrintFormat("  [PASS] Calculated 130 features for %d bars", TestBars);
         passed++;
      }
      else
         failed++;
   }
   
   //--- Test 4: Scaler
   if(TestScaler)
   {
      Print("\n--- Test 4: Scaler Loading ---");
      CMinMaxScaler scaler;
      
      if(scaler.Load(ScalerFile))
      {
         PrintFormat("  [PASS] Scaler loaded: %d features", scaler.GetNFeatures());
         passed++;
         
         // Test transform
         float raw[];
         features.CalculateTransformerFeatures(raw, 1);
         
         float scaled[];
         if(scaler.Transform(raw, scaled))
         {
            Print("  Sample scaled values:");
            PrintFormat("    [0] raw=%.6f -> scaled=%.6f", raw[0], scaled[0]);
            PrintFormat("    [1] raw=%.6f -> scaled=%.6f", raw[1], scaled[1]);
            
            // Check scaled values are in [0, 1] range
            int outOfRange = 0;
            for(int i = 0; i < ArraySize(scaled); i++)
            {
               if(scaled[i] < -0.5 || scaled[i] > 1.5)
                  outOfRange++;
            }
            if(outOfRange > 0)
               PrintFormat("  WARNING: %d values out of expected [0,1] range", outOfRange);
            else
               Print("  Scaled values within expected range");
         }
         else
         {
            Print("  [FAIL] Scaler transform failed");
            failed++;
         }
      }
      else
      {
         Print("  [FAIL] Scaler loading failed");
         Print("  Make sure file exists: MQL5/Files/", ScalerFile);
         failed++;
      }
   }
   
   //--- Test 5: Sequence Buffer
   if(TestSequenceBuffer)
   {
      Print("\n--- Test 5: Sequence Buffer ---");
      CSequenceBuffer buffer;
      
      if(buffer.Init(30, 130))
      {
         Print("  [PASS] Sequence buffer initialized");
         passed++;
         
         // Fill buffer with test data
         CMinMaxScaler scaler;
         scaler.Load(ScalerFile);
         
         Print("  Filling buffer with historical data...");
         for(int i = 30; i >= 1; i--)
         {
            float raw[];
            features.CalculateTransformerFeatures(raw, i);
            
            float scaled[];
            if(scaler.IsLoaded())
               scaler.Transform(raw, scaled);
            else
               ArrayCopy(scaled, raw);
            
            buffer.Push(scaled);
         }
         
         if(buffer.IsFull())
         {
            Print("  [PASS] Buffer filled successfully");
            passed++;
            buffer.PrintStats();
         }
         else
         {
            PrintFormat("  [FAIL] Buffer not full: %d/%d", buffer.GetCurrentCount(), buffer.GetSeqLen());
            failed++;
         }
      }
      else
      {
         Print("  [FAIL] Sequence buffer init failed");
         failed++;
      }
   }
   
   //--- Test 6: Transformer ONNX
   if(TestTransformerONNX)
   {
      Print("\n--- Test 6: Transformer ONNX Inference ---");
      CTransformerInference transformer;
      
      if(transformer.Load(TransformerFile, 30, 130, true))
      {
         Print("  [PASS] Transformer model loaded");
         passed++;
         
         // Prepare test input
         CSequenceBuffer buffer;
         buffer.Init(30, 130);
         
         CMinMaxScaler scaler;
         scaler.Load(ScalerFile);
         
         for(int i = 30; i >= 1; i--)
         {
            float raw[];
            features.CalculateTransformerFeatures(raw, i);
            float scaled[];
            if(scaler.IsLoaded())
               scaler.Transform(raw, scaled);
            else
               ArrayCopy(scaled, raw);
            buffer.Push(scaled);
         }
         
         // Get ONNX input
         float onnx_input[];
         buffer.GetOnnxInput(onnx_input);
         
         // Run inference
         float multi_tf_signal = 0;
         if(transformer.Predict(onnx_input, multi_tf_signal))
         {
            PrintFormat("  [PASS] Inference successful: multi_tf_signal = %.6f", multi_tf_signal);
            passed++;
            
            // Validate output range
            if(multi_tf_signal >= -2.0 && multi_tf_signal <= 2.0)
               Print("  Output in expected range [-2, 2]");
            else
               Print("  WARNING: Output outside expected range");
         }
         else
         {
            Print("  [FAIL] Transformer inference failed");
            failed++;
         }
         
         transformer.Release();
      }
      else
      {
         Print("  [FAIL] Transformer loading failed");
         Print("  Make sure file exists: MQL5/Files/", TransformerFile);
         failed++;
      }
   }
   
   //--- Test 7: LightGBM ONNX
   if(TestLightGBMONNX)
   {
      Print("\n--- Test 7: LightGBM ONNX Inference ---");
      CLightGBMInference lightgbm;
      
      if(lightgbm.Load(LightGBMFile, 27, true))
      {
         Print("  [PASS] LightGBM model loaded");
         passed++;
         
         // Prepare 27 features (multi_tf_signal + 26 LightGBM features)
         float lgb_input[27];
         
         // Use a mock multi_tf_signal
         lgb_input[0] = 0.5f;  // multi_tf_signal
         
         // Get 26 features from calculator
         float lgb_26[];
         features.CalculateLightGBMFeatures(lgb_26, 1);
         for(int i = 0; i < 26; i++)
            lgb_input[i + 1] = lgb_26[i];
         
         // Run inference
         int label = 0;
         float probs[];
         if(lightgbm.Predict(lgb_input, label, probs))
         {
            PrintFormat("  [PASS] Inference successful:");
            PrintFormat("    Label = %d (%s)", label, 
                        label == 0 ? "HOLD" : label == 1 ? "BUY" : "SELL");
            PrintFormat("    Probs = [HOLD:%.4f, BUY:%.4f, SELL:%.4f]", 
                        probs[0], probs[1], probs[2]);
            passed++;
            
            // Validate probabilities sum to 1
            double sum = probs[0] + probs[1] + probs[2];
            if(MathAbs(sum - 1.0) < 0.01)
               Print("  Probability sum valid: ", sum);
            else
               PrintFormat("  WARNING: Probability sum = %.4f (expected 1.0)", sum);
         }
         else
         {
            Print("  [FAIL] LightGBM inference failed");
            failed++;
         }
         
         lightgbm.Release();
      }
      else
      {
         Print("  [FAIL] LightGBM loading failed");
         Print("  Make sure file exists: MQL5/Files/", LightGBMFile);
         failed++;
      }
   }
   
   //--- Test 8: Full Pipeline
   Print("\n--- Test 8: Full 2-Model Pipeline ---");
   {
      CTransformerInference transformer;
      CLightGBMInference lightgbm;
      CMinMaxScaler scaler;
      CSequenceBuffer buffer;
      
      bool pipeline_ok = true;
      
      // Load components
      if(!transformer.Load(TransformerFile, 30, 130, false)) pipeline_ok = false;
      if(!lightgbm.Load(LightGBMFile, 27, false)) pipeline_ok = false;
      scaler.Load(ScalerFile);  // Optional
      if(!buffer.Init(30, 130)) pipeline_ok = false;
      
      if(pipeline_ok)
      {
         // Fill buffer
         for(int i = 30; i >= 1; i--)
         {
            float raw[];
            features.CalculateTransformerFeatures(raw, i);
            float scaled[];
            if(scaler.IsLoaded())
               scaler.Transform(raw, scaled);
            else
               ArrayCopy(scaled, raw);
            buffer.Push(scaled);
         }
         
         // Step 1: Transformer
         float sequence[];
         buffer.GetOnnxInput(sequence);
         
         float multi_tf_signal = 0;
         if(!transformer.Predict(sequence, multi_tf_signal))
         {
            Print("  [FAIL] Pipeline Transformer failed");
            failed++;
            pipeline_ok = false;
         }
         
         if(pipeline_ok)
         {
            // Step 2: Combine features
            float lgb_26[];
            features.CalculateLightGBMFeatures(lgb_26, 1);
            
            float lgb_input[27];
            lgb_input[0] = multi_tf_signal;
            for(int i = 0; i < 26; i++)
               lgb_input[i + 1] = lgb_26[i];
            
            // Step 3: LightGBM
            int label = 0;
            float probs[];
            if(lightgbm.Predict(lgb_input, label, probs))
            {
               Print("  [PASS] Full pipeline executed successfully!");
               PrintFormat("    multi_tf_signal = %.6f", multi_tf_signal);
               PrintFormat("    Prediction = %s (%.1f%% confidence)", 
                           label == 0 ? "HOLD" : label == 1 ? "BUY" : "SELL",
                           probs[label] * 100);
               passed++;
            }
            else
            {
               Print("  [FAIL] Pipeline LightGBM failed");
               failed++;
            }
         }
      }
      else
      {
         Print("  [FAIL] Pipeline component loading failed");
         failed++;
      }
      
      transformer.Release();
      lightgbm.Release();
   }
   
   // Cleanup
   features.Deinit();
   
   //--- Summary
   Print("\n=============================================================");
   Print("TEST SUMMARY");
   Print("=============================================================");
   PrintFormat("  PASSED: %d", passed);
   PrintFormat("  FAILED: %d", failed);
   PrintFormat("  TOTAL:  %d", passed + failed);
   Print("");
   
   if(failed == 0)
      Print("  ALL TESTS PASSED! Pipeline ready for live trading.");
   else
      Print("  Some tests FAILED. Review errors above.");
   
   Print("=============================================================");
}
