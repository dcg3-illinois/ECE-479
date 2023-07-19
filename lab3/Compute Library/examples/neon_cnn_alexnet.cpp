/*
 * Copyright (c) 2016-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class NEONCNNExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        ARM_COMPUTE_UNUSED(argc);
        ARM_COMPUTE_UNUSED(argv);

        // Create memory manager components
        // We need 2 memory managers: 1 for handling the tensors within the functions (mm_layers) and 1 for handling the input and output tensors of the functions (mm_transitions))
        auto lifetime_mgr0  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto lifetime_mgr1  = std::make_shared<BlobLifetimeManager>();                           // Create lifetime manager
        auto pool_mgr0      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto pool_mgr1      = std::make_shared<PoolManager>();                                   // Create pool manager
        auto mm_layers      = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0); // Create the memory manager
        auto mm_transitions = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1); // Create the memory manager

        // The weights and biases tensors should be initialized with the values inferred with the training

        // Set memory manager where allowed to manage internal memory requirements
        conv1   = std::make_unique<NEConvolutionLayer>(mm_layers);
        conv2   = std::make_unique<NEConvolutionLayer>(mm_layers);
        conv3   = std::make_unique<NEConvolutionLayer>(mm_layers);
        conv4   = std::make_unique<NEConvolutionLayer>(mm_layers);
        conv5   = std::make_unique<NEConvolutionLayer>(mm_layers);
        fc6     = std::make_unique<NEFullyConnectedLayer>(mm_layers);
        fc7     = std::make_unique<NEFullyConnectedLayer>(mm_layers);
        fc8     = std::make_unique<NEFullyConnectedLayer>(mm_layers);

        softmax = std::make_unique<NESoftmaxLayer>(mm_layers);

        /* [Initialize tensors] */

        // Initialize src tensor
        constexpr unsigned int width_src_image  = 256;
        constexpr unsigned int height_src_image = 256;
        constexpr unsigned int ifm_src_img      = 1;

        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

        // Initialize tensors of conv1
        constexpr unsigned int kernel_x_conv1 = 11;
        constexpr unsigned int kernel_y_conv1 = 11;
        constexpr unsigned int ofm_conv1      = 96;

        const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, src_shape.z(), ofm_conv1);
        const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
        const TensorShape out_shape_conv1(src_shape.x(), src_shape.y(), weights_shape_conv1[3]);

        weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
        biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
        out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));
        
        // NPYLoader npy_weights1;
        // NPYLoader npy_biases1;
        // npy_weights1.init_tensor(weights1, DataType::F32);
        // npy_biases1.init_tensor(biases1, DataType::F32);

        // Initialize tensor of act1
        out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

        // Initialize tensor of pool1
        TensorShape out_shape_pool1 = out_shape_conv1;
        out_shape_pool1.set(0, out_shape_pool1.x() / 2);
        out_shape_pool1.set(1, out_shape_pool1.y() / 2);
        out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));

        // Initialize tensors of conv2
        constexpr unsigned int kernel_x_conv2 = 5;
        constexpr unsigned int kernel_y_conv2 = 5;
        constexpr unsigned int ofm_conv2      = 256;

        const TensorShape weights_shape_conv2(kernel_x_conv2, kernel_y_conv2, out_shape_pool1.z(), ofm_conv2);

        const TensorShape biases_shape_conv2(weights_shape_conv2[3]);
        const TensorShape out_shape_conv2(out_shape_pool1.x(), out_shape_pool1.y(), weights_shape_conv2[3]);

        weights2.allocator()->init(TensorInfo(weights_shape_conv2, 1, DataType::F32));
        biases2.allocator()->init(TensorInfo(biases_shape_conv2, 1, DataType::F32));
        out_conv2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));

        // Initialize tensor of act2
        out_act2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));

        // Initialize tensor of pool2
        TensorShape out_shape_pool2 = out_shape_conv2;
        out_shape_pool2.set(0, out_shape_pool2.x() / 2);
        out_shape_pool2.set(1, out_shape_pool2.y() / 2);
        out_pool2.allocator()->init(TensorInfo(out_shape_pool2, 1, DataType::F32));

        // Initialize tensors of conv3
        constexpr unsigned int kernel_x_conv3 = 3;
        constexpr unsigned int kernel_y_conv3 = 3;
        constexpr unsigned int ofm_conv3      = 384;

        const TensorShape weights_shape_conv3(kernel_x_conv3, kernel_y_conv3, out_shape_pool2.z(), ofm_conv3);

        const TensorShape biases_shape_conv3(weights_shape_conv3[3]);
        const TensorShape out_shape_conv3(out_shape_pool2.x(), out_shape_pool2.y(), weights_shape_conv3[3]);

        weights3.allocator()->init(TensorInfo(weights_shape_conv3, 1, DataType::F32));
        biases3.allocator()->init(TensorInfo(biases_shape_conv3, 1, DataType::F32));
        out_conv3.allocator()->init(TensorInfo(out_shape_conv3, 1, DataType::F32));

        // Initialize tensor of act3
        out_act3.allocator()->init(TensorInfo(out_shape_conv3, 1, DataType::F32));
        
        // Initialize tensors of conv4
        constexpr unsigned int kernel_x_conv4 = 3;
        constexpr unsigned int kernel_y_conv4 = 3;
        constexpr unsigned int ofm_conv4      = 384;

        const TensorShape weights_shape_conv4(kernel_x_conv4, kernel_y_conv4, out_shape_conv3.z(), ofm_conv4);

        const TensorShape biases_shape_conv4(weights_shape_conv4[3]);
        const TensorShape out_shape_conv4(out_shape_conv3.x(), out_shape_conv3.y(), weights_shape_conv4[3]);

        weights4.allocator()->init(TensorInfo(weights_shape_conv4, 1, DataType::F32));
        biases4.allocator()->init(TensorInfo(biases_shape_conv4, 1, DataType::F32));
        out_conv4.allocator()->init(TensorInfo(out_shape_conv4, 1, DataType::F32));

        // Initialize tensor of act4
        out_act4.allocator()->init(TensorInfo(out_shape_conv4, 1, DataType::F32));

        // Initialize tensors of conv5
        constexpr unsigned int kernel_x_conv5 = 3;
        constexpr unsigned int kernel_y_conv5 = 3;
        constexpr unsigned int ofm_conv5      = 256;

        const TensorShape weights_shape_conv5(kernel_x_conv5, kernel_y_conv5, out_shape_conv4.z(), ofm_conv5);

        const TensorShape biases_shape_conv5(weights_shape_conv5[3]);
        const TensorShape out_shape_conv5(out_shape_conv4.x(), out_shape_conv4.y(), weights_shape_conv5[3]);

        weights5.allocator()->init(TensorInfo(weights_shape_conv5, 1, DataType::F32));
        biases5.allocator()->init(TensorInfo(biases_shape_conv5, 1, DataType::F32));
        out_conv5.allocator()->init(TensorInfo(out_shape_conv5, 1, DataType::F32));

        // Initialize tensor of act5
        out_act5.allocator()->init(TensorInfo(out_shape_conv5, 1, DataType::F32));

        // Initialize tensor of pool5
        TensorShape out_shape_pool5 = out_shape_conv5;
        out_shape_pool5.set(0, out_shape_pool5.x() / 2);
        out_shape_pool5.set(1, out_shape_pool5.y() / 2);
        out_pool5.allocator()->init(TensorInfo(out_shape_pool5, 1, DataType::F32));

        // Initialize tensor of fc6
        constexpr unsigned int num_nodes = 4096;

        const TensorShape weights_shape_fc6(out_shape_pool5.x() * out_shape_pool5.y() * out_shape_pool5.z(), num_nodes);
        const TensorShape biases_shape_fc6(num_nodes);
        const TensorShape out_shape_fc6(num_nodes);

        weights6.allocator()->init(TensorInfo(weights_shape_fc6, 1, DataType::F32));
        biases6.allocator()->init(TensorInfo(biases_shape_fc6, 1, DataType::F32));
        out_fc6.allocator()->init(TensorInfo(out_shape_fc6, 1, DataType::F32));

        // Initialize tensor of act2
        out_act6.allocator()->init(TensorInfo(out_shape_fc6, 1, DataType::F32));

        // Initialize tensor of fc7
        const TensorShape weights_shape_fc7(out_shape_fc6.x() * out_shape_fc6.y() * out_shape_fc6.z(), num_nodes);
        const TensorShape biases_shape_fc7(num_nodes);
        const TensorShape out_shape_fc7(num_nodes);

        weights7.allocator()->init(TensorInfo(weights_shape_fc7, 1, DataType::F32));
        biases7.allocator()->init(TensorInfo(biases_shape_fc7, 1, DataType::F32));
        out_fc7.allocator()->init(TensorInfo(out_shape_fc7, 1, DataType::F32));

        // Initialize tensor of act7
        out_act7.allocator()->init(TensorInfo(out_shape_fc7, 1, DataType::F32));

        // Initialize tensor of fc8
        constexpr unsigned int num_labels = 1000;

        const TensorShape weights_shape_fc8(out_shape_fc7.x() * out_shape_fc7.y() * out_shape_fc7.z(), num_labels);
        const TensorShape biases_shape_fc8(num_labels);
        const TensorShape out_shape_fc8(num_labels);

        weights8.allocator()->init(TensorInfo(weights_shape_fc8, 1, DataType::F32));
        biases8.allocator()->init(TensorInfo(biases_shape_fc8, 1, DataType::F32));
        out_fc8.allocator()->init(TensorInfo(out_shape_fc8, 1, DataType::F32));

        // Initialize tensor of softmax
        const TensorShape out_shape_softmax(out_shape_fc8.x());
        out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));

        constexpr auto data_layout = DataLayout::NCHW;

        /* -----------------------End: [Initialize tensors] */

        /* [Configure functions] */
        conv1->configure(&src, &weights1, &biases1, &out_conv1, PadStrideInfo(4 /* stride_x */, 4 /* stride_y */, 0 /* pad_x */, 0 /* pad_y */));
        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        pool1.configure(&out_act1, &out_pool1, PoolingLayerInfo(PoolingType::MAX, 3, data_layout, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

        conv2->configure(&out_pool1, &weights2, &biases2, &out_conv2, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 2 /* pad_x */, 2 /* pad_y */));
        act2.configure(&out_conv2, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        pool2.configure(&out_act2, &out_pool2, PoolingLayerInfo(PoolingType::MAX, 3, data_layout, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));
        
        conv3->configure(&out_pool2, &weights3, &biases3, &out_conv3, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1 /* pad_x */, 1 /* pad_y */));
        act3.configure(&out_conv3, &out_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        conv4->configure(&out_act3, &weights4, &biases4, &out_conv4, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1 /* pad_x */, 1 /* pad_y */));
        act4.configure(&out_conv4, &out_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        conv5->configure(&out_act4, &weights5, &biases5, &out_conv5, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1 /* pad_x */, 1 /* pad_y */));
        act5.configure(&out_conv5, &out_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        pool5.configure(&out_act2, &out_pool2, PoolingLayerInfo(PoolingType::MAX, 3, data_layout, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */)));

        fc6->configure(&out_pool5, &weights6, &biases6, &out_fc6);
        act6.configure(&out_fc6, &out_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        fc7->configure(&out_act6, &weights7, &biases7, &out_fc7);
        act7.configure(&out_fc7, &out_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        fc8->configure(&out_act7, &weights8, &biases8, &out_fc8);
        softmax->configure(&out_fc8, &out_softmax);

        /* -----------------------End: [Configure functions] */

        /*[ Add tensors to memory manager ]*/

        // We need 2 memory groups for handling the input and output
        // We call explicitly allocate after manage() in order to avoid overlapping lifetimes
        memory_group0 = std::make_unique<MemoryGroup>(mm_transitions);
        memory_group1 = std::make_unique<MemoryGroup>(mm_transitions);

        memory_group0->manage(&out_conv1);
        out_conv1.allocator()->allocate();
        memory_group1->manage(&out_act1);
        out_act1.allocator()->allocate();
        memory_group0->manage(&out_pool1);
        out_pool1.allocator()->allocate();

        memory_group1->manage(&out_conv2);
        out_conv2.allocator()->allocate();
        memory_group0->manage(&out_act2);
        out_act2.allocator()->allocate();
        memory_group1->manage(&out_pool2);
        out_pool2.allocator()->allocate();

        memory_group0->manage(&out_conv3);
        out_conv3.allocator()->allocate();
        memory_group1->manage(&out_act3);
        out_act3.allocator()->allocate();

        memory_group0->manage(&out_conv4);
        out_conv4.allocator()->allocate();
        memory_group1->manage(&out_act4);
        out_act4.allocator()->allocate();

        memory_group0->manage(&out_conv5);
        out_conv5.allocator()->allocate();
        memory_group1->manage(&out_act5);
        out_act5.allocator()->allocate();
        memory_group0->manage(&out_pool5);
        out_pool5.allocator()->allocate();

        memory_group1->manage(&out_fc6);
        out_fc6.allocator()->allocate();
        memory_group0->manage(&out_act6);
        out_act6.allocator()->allocate();

        memory_group1->manage(&out_fc7);
        out_fc7.allocator()->allocate();
        memory_group0->manage(&out_act7);
        out_act7.allocator()->allocate();

        memory_group1->manage(&out_fc8);
        out_fc8.allocator()->allocate();
        memory_group0->manage(&out_softmax);
        out_softmax.allocator()->allocate();

        /* -----------------------End: [ Add tensors to memory manager ] */

        /* [Allocate tensors] */

        // Now that the padding requirements are known we can allocate all tensors
        src.allocator()->allocate();
        weights1.allocator()->allocate();
        weights2.allocator()->allocate();
        weights3.allocator()->allocate();
        weights4.allocator()->allocate();
        weights5.allocator()->allocate();
        weights6.allocator()->allocate();
        weights7.allocator()->allocate();
        weights8.allocator()->allocate();

        biases1.allocator()->allocate();
        biases2.allocator()->allocate();
        biases3.allocator()->allocate();
        biases4.allocator()->allocate();
        biases5.allocator()->allocate();
        biases6.allocator()->allocate();
        biases7.allocator()->allocate();
        biases8.allocator()->allocate();

        /* -----------------------End: [Allocate tensors] */

        // Populate the layers manager. (Validity checks, memory allocations etc)
        mm_layers->populate(allocator, 1 /* num_pools */);

        // Populate the transitions manager. (Validity checks, memory allocations etc)
        mm_transitions->populate(allocator, 2 /* num_pools */);

        return true;
    }
    void do_run() override
    {
        // Acquire memory for the memory groups
        memory_group0->acquire();
        memory_group1->acquire();

        conv1->run();
        act1.run();
        pool1.run();

        conv2->run();
        act2.run();
        pool2.run();

        conv3->run();
        act3.run();

        conv4->run();
        act4.run();

        conv5->run();
        act5.run();
        pool5.run();

        fc6->run();
        act6.run();

        fc7->run();
        act7.run();

        fc8->run();
        softmax->run();

        // Release memory
        memory_group0->release();
        memory_group1->release();
    }

private:
    // The src tensor should contain the input image
    Tensor src{};

    // Intermediate tensors used
    Tensor weights1{};
    Tensor weights2{};
    Tensor weights3{};
    Tensor weights4{};
    Tensor weights5{};
    Tensor weights6{};
    Tensor weights7{};
    Tensor weights8{};

    Tensor biases1{};
    Tensor biases2{};
    Tensor biases3{};
    Tensor biases4{};
    Tensor biases5{};
    Tensor biases6{};
    Tensor biases7{};
    Tensor biases8{};

    Tensor out_conv1{};
    Tensor out_conv2{};
    Tensor out_conv3{};
    Tensor out_conv4{};
    Tensor out_conv5{};
    Tensor out_fc6{};
    Tensor out_fc7{};
    Tensor out_fc8{};
    
    Tensor out_act1{};
    Tensor out_act2{};
    Tensor out_act3{};
    Tensor out_act4{};
    Tensor out_act5{};
    Tensor out_act6{};
    Tensor out_act7{};
    Tensor out_softmax{};

    Tensor out_pool1{};
    Tensor out_pool2{};
    Tensor out_pool5{};

    Tensor out_norm1{};
    Tensor out_norm2{};

    // Allocator
    Allocator allocator{};

    // Memory groups
    std::unique_ptr<MemoryGroup> memory_group0{};
    std::unique_ptr<MemoryGroup> memory_group1{};

    // Layers
    std::unique_ptr<NEConvolutionLayer>    conv1{};
    std::unique_ptr<NEConvolutionLayer>    conv2{};
    std::unique_ptr<NEConvolutionLayer>    conv3{};
    std::unique_ptr<NEConvolutionLayer>    conv4{};
    std::unique_ptr<NEConvolutionLayer>    conv5{};
    std::unique_ptr<NEFullyConnectedLayer> fc6{};
    std::unique_ptr<NEFullyConnectedLayer> fc7{};
    std::unique_ptr<NEFullyConnectedLayer> fc8{};
    std::unique_ptr<NESoftmaxLayer>        softmax{};
    NEPoolingLayer                         pool1{};
    NEPoolingLayer                         pool2{};
    NEPoolingLayer                         pool5{};
    NEActivationLayer                      act1{};
    NEActivationLayer                      act2{};
    NEActivationLayer                      act3{};
    NEActivationLayer                      act4{};
    NEActivationLayer                      act5{};
    NEActivationLayer                      act6{};
    NEActivationLayer                      act7{};
    // NENormalizationLayer                   norm1{};
    // NENormalizationLayer                   norm2{};
};

/** Main program for cnn test
 *
 * The example implements the following CNN architecture:
 *
 * Input -> conv0:5x5 -> act0:relu -> pool:2x2 -> conv1:3x3 -> act1:relu -> pool:2x2 -> fc0 -> act2:relu -> softmax
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONCNNExample>(argc, argv);
}
