#include <c_api.h> // TensorFlow C API header
#include <c_api_experimental.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

#define MAGICKCORE_QUANTUM_DEPTH 16
#define MAGICKCORE_HDRI_ENABLE 0

// https://imagemagick.org/api/magick-image.php
#include <wand/magick_wand.h>

static const char *CodeToString(TF_Code code);
template <typename T>
static TF_Tensor *CreateTensor(TF_DataType data_type, const std::vector<int64_t> &dims, const std::vector<T> &data);
static void DeallocateBuffer(void *data, size_t);
static TF_Buffer *ReadBufferFromFile(const char *file);

int main()
{
    MagickWand *mw = NULL;
    MagickWandGenesis();
    mw = NewMagickWand();
    MagickReadImage(mw, "../assets/sample.jpg");
    MagickScaleImage(mw, 512, 288);
    std::vector<std::uint8_t> pixels;
    pixels.resize(512 * 288 * 3);
    MagickExportImagePixels(mw, 0, 0, 512, 288, "RGB", CharPixel, pixels.data());
    MagickWandTerminus();

    std::cout << TF_Version() << std::endl;

    TF_Buffer *graph_buffer = ReadBufferFromFile("../models/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb");
    if (graph_buffer == nullptr)
    {
        std::cout << "Can't read buffer from file" << std::endl;
        return 1;
    }

    TF_Graph *graph = TF_NewGraph();
    TF_Status *status = TF_NewStatus();
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(graph, graph_buffer, opts, status);
    std::cout << "Status Code: " << TF_GetCode(status) << " - " << CodeToString(TF_GetCode(status)) << std::endl;
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graph_buffer);

    if (TF_GetCode(status) != TF_OK)
    {
        TF_DeleteGraph(graph);
        std::cout << "Can't import GraphDef" << std::endl;
        return 2;
    }

    std::cout << "Load graph success" << std::endl;

    auto input_op = TF_Output{TF_GraphOperationByName(graph, "ImageTensor"), 0};
    if (input_op.oper == nullptr)
    {
        std::cout << "Can't init input_op" << std::endl;
        return 3;
    }

    auto output_op = TF_Output{TF_GraphOperationByName(graph, "SemanticPredictions"), 0};
    if (output_op.oper == nullptr)
    {
        std::cout << "Can't init output_op" << std::endl;
        return 4;
    }

    const std::vector<std::int64_t> input_dims = {1, 512, 288, 3};
    TF_Tensor *input_tensor = CreateTensor<uint8_t>(TF_UINT8, input_dims, pixels);

    TF_Tensor *output_tensor = nullptr;

    TF_SessionOptions *options = TF_NewSessionOptions();
    // TF_EnableXLACompilation(options, true);

    TF_Session *session = TF_NewSession(graph, options, status);
    std::cout << "Status Code: " << TF_GetCode(status) << " - " << CodeToString(TF_GetCode(status)) << std::endl;
    TF_DeleteSessionOptions(options);
    if (TF_GetCode(status) != TF_OK)
    {
        return 5;
    }

    TF_SessionRun(session,
                  nullptr,                       // Run options.
                  &input_op, &input_tensor, 1,   // Input tensors, input tensor values, number of inputs.
                  &output_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                  nullptr, 0,                    // Target operations, number of targets.
                  nullptr,                       // Run metadata.
                  status                         // Output status.
    );

    std::cout << "Status Code: " << TF_GetCode(status) << " - " << CodeToString(TF_GetCode(status)) << std::endl;

    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error run session";
        return 6;
    }


    std::cout << "TF_TensorByteSize: " << TF_TensorByteSize(output_tensor) << std::endl;
    std::cout << "    TF_TensorType: " << TF_TensorType(output_tensor) << std::endl;
    std::cout << "       TF_NumDims: " << TF_NumDims(output_tensor) << std::endl;
    int num_dims = TF_NumDims(output_tensor);
    for (int i = 0; i < num_dims; ++i)
        std::cout << "           TF_Dim: " << i << " " << TF_Dim(output_tensor, i) << std::endl;

    int64_t *data = static_cast<std::int64_t *>(TF_TensorData(output_tensor));

    std::vector<std::int64_t> output;
    output.assign(data, data + 512 * 288);

    for (auto i=0; i < 512 * 288; ++i) 
    {
        // std::cout << data[i] << " ";
    }

    std::cout << "     Raw max_value: " << *std::max_element(data, data + 512 * 288) << std::endl;
    std::cout << "  Output max_value: " << *std::max_element(output.begin(), output.end()) << std::endl;

    TF_CloseSession(session, status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error close session";
        return 7;
    }

    TF_DeleteSession(session, status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error delete session";
        return 8;
    }

    TF_DeleteStatus(status);
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);

    return 0;
}

static const char *CodeToString(TF_Code code)
{
    switch (code)
    {
    case TF_OK:
        return "TF_OK";
    case TF_CANCELLED:
        return "TF_CANCELLED";
    case TF_UNKNOWN:
        return "TF_UNKNOWN";
    case TF_INVALID_ARGUMENT:
        return "TF_INVALID_ARGUMENT";
    case TF_DEADLINE_EXCEEDED:
        return "TF_DEADLINE_EXCEEDED";
    case TF_NOT_FOUND:
        return "TF_NOT_FOUND";
    case TF_ALREADY_EXISTS:
        return "TF_ALREADY_EXISTS";
    case TF_PERMISSION_DENIED:
        return "TF_PERMISSION_DENIED";
    case TF_UNAUTHENTICATED:
        return "TF_UNAUTHENTICATED";
    case TF_RESOURCE_EXHAUSTED:
        return "TF_RESOURCE_EXHAUSTED";
    case TF_FAILED_PRECONDITION:
        return "TF_FAILED_PRECONDITION";
    case TF_ABORTED:
        return "TF_ABORTED";
    case TF_OUT_OF_RANGE:
        return "TF_OUT_OF_RANGE";
    case TF_UNIMPLEMENTED:
        return "TF_UNIMPLEMENTED";
    case TF_INTERNAL:
        return "TF_INTERNAL";
    case TF_UNAVAILABLE:
        return "TF_UNAVAILABLE";
    case TF_DATA_LOSS:
        return "TF_DATA_LOSS";
    default:
        return "Unknown";
    }
}

template <typename T>
static TF_Tensor *CreateTensor(TF_DataType data_type, const std::vector<int64_t> &dims, const std::vector<T> &data)
{

    auto data_size = sizeof(T);
    for (auto i : dims)
    {
        data_size *= i;
    }

    auto tensor = TF_AllocateTensor(data_type, dims.data(), static_cast<int>(dims.size()), data_size);
    std::cout << "Allocating " << data_size << " bytes" << std::endl;

    if (tensor != nullptr && TF_TensorData(tensor) != nullptr)
    {
        std::cout << "TF_TensorByteSize(tensor) " << TF_TensorByteSize(tensor) << std::endl;
        size_t cp_len = std::min(data_size, TF_TensorByteSize(tensor));
        std::cout << "Copying " << cp_len << " bytes to tensor" << std::endl;
        std::memcpy(TF_TensorData(tensor), data.data(), cp_len);
    }
    else
    {
        std::cout << "Wrong creat tensor" << std::endl;
        return NULL;
    }

    if (TF_TensorType(tensor) != data_type)
    {
        std::cout << "Wrong tensor type" << std::endl;
        return NULL;
    }

    if (TF_NumDims(tensor) != static_cast<int>(dims.size()))
    {
        std::cout << "Wrong number of dimensions" << std::endl;
        return NULL;
    }

    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        if (TF_Dim(tensor, static_cast<int>(i)) != dims[i])
        {
            std::cout << "Wrong dimension size for dim: " << i << std::endl;
            return NULL;
        }
    }

    if (TF_TensorByteSize(tensor) != data_size)
    {
        std::cout << "Wrong tensor byte size" << std::endl;
        return NULL;
    }

    auto tensor_data = static_cast<T *>(TF_TensorData(tensor));

    if (tensor_data == nullptr)
    {
        std::cout << "Wrong data tensor" << std::endl;
        return NULL;
    }

    for (std::size_t i = 0; i < data.size(); ++i)
    {
        if (tensor_data[i] != data[i])
        {
            std::cout << "Element: " << i << " does not match" << std::endl;
            return NULL;
        }
    }

    std::cout << "Success allocate tensor" << std::endl;

    return tensor;
}

static void DeallocateBuffer(void *data, size_t)
{
    std::free(data);
}

static TF_Buffer *ReadBufferFromFile(const char *file)
{
    std::ifstream f(file, std::ios::binary);

    if (f.fail() || !f.is_open())
    {
        f.close();
        return nullptr;
    }

    f.seekg(0, std::ios::end);
    auto fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    if (fsize < 1)
    {
        f.close();
        return nullptr;
    }

    auto data = static_cast<char *>(std::malloc(fsize));
    f.read(data, fsize);

    auto buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;

    f.close();
    return buf;
}