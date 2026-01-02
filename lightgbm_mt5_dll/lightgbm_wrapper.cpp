// lightgbm_wrapper.cpp
// DLL wrapper for LightGBM model to be called from MT5

#include <windows.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

// LightGBM C API
#include "LightGBM/c_api.h"

// Global model handle
static BoosterHandle g_booster = nullptr;

// DLL Export macro
#define DLL_EXPORT extern "C" __declspec(dllexport)

/**
 * Load LightGBM model from file
 * @param model_path Path to the LightGBM model file (.txt format)
 * @return 1 on success, 0 on failure
 */
DLL_EXPORT int LoadModel(const wchar_t* model_path)
{
    // Convert wide string to narrow string
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, model_path, -1, NULL, 0, NULL, NULL);
    std::string path(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, model_path, -1, &path[0], size_needed, NULL, NULL);
    path.resize(size_needed - 1); // Remove null terminator

    // Load model
    int num_iterations = 0;
    int result = LGBM_BoosterCreateFromModelfile(
        path.c_str(),
        &num_iterations,
        &g_booster
    );

    return (result == 0) ? 1 : 0;
}

/**
 * Predict using loaded LightGBM model
 * @param features Input feature array (26 features)
 * @param probs Output probability array (3 classes: SHORT, HOLD, LONG)
 * @return 1 on success, 0 on failure
 */
DLL_EXPORT int Predict(double* features, double* probs)
{
    if (g_booster == nullptr) {
        return 0; // Model not loaded
    }

    // Prepare input data
    const int num_features = 26;
    const int num_classes = 3;

    // Predict
    int64_t out_len;
    int result = LGBM_BoosterPredictForMat(
        g_booster,
        features,           // Input features
        C_API_DTYPE_FLOAT64, // Data type
        1,                  // Number of rows (batch size = 1)
        num_features,       // Number of features
        1,                  // is_row_major
        C_API_PREDICT_NORMAL, // Predict type
        0,                  // start_iteration
        -1,                 // num_iteration (-1 = use all)
        "",                 // parameter
        &out_len,           // output length
        probs               // output probabilities
    );

    return (result == 0) ? 1 : 0;
}

/**
 * Unload model and free resources
 * @return 1 on success, 0 on failure
 */
DLL_EXPORT int UnloadModel()
{
    if (g_booster != nullptr) {
        int result = LGBM_BoosterFree(g_booster);
        g_booster = nullptr;
        return (result == 0) ? 1 : 0;
    }
    return 1;
}

/**
 * Get model information
 * @param info_buffer Buffer to store info string
 * @param buffer_size Size of buffer
 * @return 1 on success, 0 on failure
 */
DLL_EXPORT int GetModelInfo(wchar_t* info_buffer, int buffer_size)
{
    if (g_booster == nullptr) {
        return 0;
    }

    int num_features = 0;
    int num_classes = 0;
    int num_trees = 0;

    // Get number of features
    LGBM_BoosterGetNumFeature(g_booster, &num_features);

    // Get number of classes
    LGBM_BoosterGetNumClasses(g_booster, &num_classes);

    // Create info string
    std::wstring info = L"Features: " + std::to_wstring(num_features) +
                        L", Classes: " + std::to_wstring(num_classes);

    // Copy to buffer
    wcsncpy_s(info_buffer, buffer_size, info.c_str(), _TRUNCATE);

    return 1;
}

// DLL entry point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
