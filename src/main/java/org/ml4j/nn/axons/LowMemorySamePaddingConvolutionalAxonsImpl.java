/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.axons;

import java.util.Optional;
import java.util.function.Supplier;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.jblas.FloatArray;
import org.ml4j.jblas.FloatArrayMatrix;
import org.ml4j.jblas.JBlasRowMajorMatrixOptimised;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Prototype experimental implementation of k2r-aa convolutional method from 
 * "Low-memory GEMM-based convolution algorithms for deep neural networks"  ( https://arxiv.org/pdf/1709.03395.pdf )
 * 
 * @author Michael Lavelle
 *
 */
public class LowMemorySamePaddingConvolutionalAxonsImpl implements ConvolutionalAxons {

	private static final Logger LOGGER = LoggerFactory.getLogger(LowMemorySamePaddingConvolutionalAxonsImpl.class);

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;
	private Axons3DConfig config;
	private AxonWeights weights;

	public LowMemorySamePaddingConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons, AxonWeights weights,
			Axons3DConfig config) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
		this.weights = weights;

	}

	public LowMemorySamePaddingConvolutionalAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config, Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		super();
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultFullyConnectedAxonWeightsInitialiser(
				new Neurons(filterWidth * filterHeight * leftNeurons.getDepth(), leftNeurons.hasBiasUnit()),
				new Neurons(rightNeurons.getDepth(), rightNeurons.hasBiasUnit()));

		Matrix initialConnectionWeights = connectionWeights == null
				? axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory)
				: connectionWeights;
		Optional<Matrix> initialLeftToRightBiases = leftToRightBiases == null
				? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(leftToRightBiases);
		Optional<Matrix> initialRightToLeftBiases = rightToLeftBiases == null
				? axonWeightsInitialiser.getInitialRightToLeftBiases(matrixFactory)
				: Optional.of(rightToLeftBiases);

		this.weights = new FullyConnectedAxonWeightsImpl(filterWidth * filterHeight * leftNeurons.getDepth(),
				rightNeurons.getDepth(), initialConnectionWeights,
				leftNeurons.hasBiasUnit() && initialLeftToRightBiases.isPresent() ? initialLeftToRightBiases.get()
						: null,
				rightNeurons.hasBiasUnit() && initialRightToLeftBiases.isPresent() ? initialRightToLeftBiases.get()
						: null);
	}

	@Override
	public void adjustAxonWeights(AxonWeightsAdjustment adjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		weights.adjustWeights(adjustment, adjustmentDirection);
	}

	@Override
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
		return rightNeurons;
	}

	private NeuronsActivation getPostDropoutInput(NeuronsActivation inputMatrix, int exampleCount,
			AxonsContext axonsContext) {

		NeuronsActivation reformatted = reformatLeftToRightInput(axonsContext.getMatrixFactory(), inputMatrix);
		reformatted.setImmutable(true);

		return reformatted;
	}

	public NeuronsActivation reformatLeftToRightInput(MatrixFactory matrixFactory,
			NeuronsActivation leftNeuronsActivation) {
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		final NeuronsActivation reformatted;
		ImageNeuronsActivation imageAct = leftNeuronsActivation.asImageNeuronsActivation(leftNeurons);
		reformatted = new NeuronsActivationImpl(
				new Neurons(leftNeurons.getDepth() * filterWidth * filterHeight, leftNeurons.hasBiasUnit()),
				imageAct.im2ColConv(matrixFactory, filterHeight, filterWidth, config.getStrideHeight(),
						config.getStrideWidth(), config.getPaddingHeight(), config.getPaddingWidth()),
				// TODO Format
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT);
		if (!imageAct.isImmutable()) {
			imageAct.close();
		}

		return reformatted;

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation inputMatrix, AxonsActivation previousRightToLeftActivation,
			AxonsContext axonsContext) {

		ImageNeuronsActivationFormat featureOrientation = ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT;

		int exampleCount = inputMatrix.getExampleCount();

		LOGGER.debug("Pushing left to right through Conv axons:" + inputMatrix.getFeatureCount() + ":"
				+ inputMatrix.getExampleCount() + ":" + featureOrientation);

		inputMatrix.setImmutable(true);

		Supplier<NeuronsActivation> postDropoutInput = () -> getPostDropoutInput(inputMatrix, exampleCount,
				axonsContext);

		Matrix kernelMatrix = weights.getConnectionWeights();
		Matrix biasMatrix = weights.getLeftToRightBiases();

		ImageNeuronsActivation convolution = performConvolution(inputMatrix, axonsContext, kernelMatrix, biasMatrix,
				leftNeurons, rightNeurons);

		return new AxonsActivationImpl(this, null, postDropoutInput, convolution);

	}

	private ImageNeuronsActivation performConvolution(NeuronsActivation inputMatrix, AxonsContext axonsContext,
			Matrix kernelMatrix, Matrix biasMatrix, Neurons3D leftNeurons, Neurons3D rightNeurons) {

		LOGGER.debug("Pushing left to right through FC axons");

		int examples = inputMatrix.getExampleCount();
		int outputChannels = rightNeurons.getDepth();
		int inputChannels = leftNeurons.getDepth();

		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;
		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int kernelWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());
		int kernelHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		int m = rightNeurons.getDepth();

		float f = ((float) kernelWidth) / ((float) (inputHeight + inputWidth));
		int delta = (int) Math.ceil((double) f);
		float[] targetData = new float[(m + 2 * delta) * inputWidth * inputHeight * examples];

		NeuronsActivation inputMat = inputMatrix;

		FloatArrayMatrix k = null;
		int targetOffset = 0;
		float[] inputData = null;
		float alpha = 1f;
		float beta = 1f;
		float[] kData = null;

		// Kernel matrix is output channels * input channels, height, width,
		Matrix kernelMatrixNew = kernelMatrix;

		Matrix inputMatDup = inputMat.getActivations(axonsContext.getMatrixFactory());

		kernelMatrixNew = kernelMatrixNew.softDup();
		kernelMatrixNew.asEditableMatrix().reshape(outputChannels * inputChannels, kernelHeight * kernelWidth);
		kernelMatrixNew = kernelMatrixNew.transpose();

		for (int h = 0; h < kernelHeight; h++) {
			for (int w = 0; w < kernelWidth; w++) {
				float[] rowS = null;
				float[] rowE = null;
				float[] colS = null;
				float[] colE = null;

				Matrix row = kernelMatrixNew.getRow(h * kernelWidth + w);
				row.asEditableMatrix().reshape(outputChannels, inputChannels);
				Matrix row2 = row;
				kData = row2.getRowByRowArray();
				k = new FloatArrayMatrix(new FloatArray(kData, 0), outputChannels, inputChannels);

				Matrix mat = inputMatDup;

				inputData = mat.getRowByRowArray();

				synchronized (inputData) {

					if (h == 0) {
						rowS = zeroRow(inputData, inputWidth, inputHeight, inputChannels, examples, inputHeight - 1);
					} else if (h == (kernelHeight - 1)) {
						rowE = zeroRow(inputData, inputWidth, inputHeight, inputChannels, examples, 0);
					}
					if (w == 0) {
						colS = zeroColumn(inputData, inputWidth, inputHeight, inputChannels, examples, inputWidth - 1);
					} else if (w == (kernelWidth - 1)) {
						colE = zeroColumn(inputData, inputWidth, inputHeight, inputChannels, examples, 0);
					}

					int z = kernelWidth / 2;

					int halfFilterHeightIndex = kernelHeight / 2;
					int halfFilterWidthIndex = kernelWidth / 2;

					int otTimesExamples = delta * (inputWidth * inputHeight * examples)
							- z * ((halfFilterHeightIndex * inputWidth) + halfFilterWidthIndex + 1) * examples;
					targetOffset = otTimesExamples + z * ((h * inputWidth) + w + 1) * examples;

					int targetOffsetAdj = (targetOffset - delta * (inputWidth * inputHeight * examples));

					int targetOffsetAdjNegated = -targetOffsetAdj;
					targetOffset = targetOffsetAdjNegated + delta * (inputWidth * inputHeight * examples);

					FloatArrayMatrix b = new FloatArrayMatrix(new FloatArray(inputData, 0), inputChannels,
							inputWidth * inputHeight * examples);

					JBlasRowMajorMatrixOptimised.gemm(alpha, k, b, beta,
							new FloatArrayMatrix(new FloatArray(targetData, targetOffset), outputChannels,
									inputWidth * inputHeight * examples));

					if (rowS != null) {
						resetRow(inputData, inputWidth, inputHeight, inputChannels, examples, inputHeight - 1, rowS);
					}
					if (rowE != null) {
						resetRow(inputData, inputWidth, inputHeight, inputChannels, examples, 0, rowE);
					}
					if (colS != null) {
						resetColumn(inputData, inputWidth, inputHeight, inputChannels, examples, inputWidth - 1, colS);
					}
					if (colE != null) {
						resetColumn(inputData, inputWidth, inputHeight, inputChannels, examples, 0, colE);
					}
				}
			}
		}

		LOGGER.debug("End left to right pushing through FC axons");

		LOGGER.debug("End Pushing left to right through Conv axons");
		// TODO
		if (leftNeurons.hasBiasUnit() && biasMatrix != null) {
			return createOutputActivationByCopyingData(axonsContext.getMatrixFactory(), targetData, inputWidth,
					inputHeight, outputChannels, examples, delta, biasMatrix);
		} else {
			return createOutputActivationByReferencingData(targetData, inputWidth, inputHeight, outputChannels,
					examples, delta, rightNeurons);
		}

	}

	private ImageNeuronsActivation createOutputActivationByReferencingData(float[] targetData, int inputWidth,
			int inputHeight, int outputChannels, int examples, int delta, Neurons3D rightNeurons) {
		Images outputImage = new MultiChannelImages(targetData, delta * inputWidth * inputHeight * examples,
				outputChannels, inputHeight, inputWidth, 0, 0, examples);
		return new ImageNeuronsActivationImpl(rightNeurons, outputImage,
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);

	}

	private ImageNeuronsActivation createOutputActivationByCopyingData(MatrixFactory matrixFactory, float[] targetData,
			int inputWidth, int inputHeight, int outputChannels, int examples, int delta, Matrix biasMatrix) {
		Matrix outFinal = matrixFactory.createMatrix(outputChannels, inputWidth * inputHeight * examples);

		float[] outFinalData = outFinal.getRowByRowArray();

		System.arraycopy(targetData, delta * inputWidth * inputHeight * examples, outFinalData, 0,
				outputChannels * inputWidth * inputHeight * examples);

		outFinal.asEditableMatrix().reshape(outputChannels * inputWidth * inputHeight, examples);
		if (biasMatrix != null) {
			outFinal.asEditableMatrix().addiColumnVector(biasMatrix);
		}

		return new ImageNeuronsActivationImpl(outFinal, rightNeurons,
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);

	}

	private float[] zeroColumn(float[] data, int width, int height, int inputChannels, int examples, int c) {
		float[] oldValues = new float[inputChannels * width * examples];
		int index = 0;
		for (int i = 0; i < inputChannels; i++) {
			for (int h = 0; h < height; h++) {
				int startInd = i * height * width * examples + h * width * examples + c * examples;
				for (int e = 0; e < examples; e++) {
					oldValues[index] = data[startInd + e];
					data[startInd + e] = 0;
					index++;
				}
			}
		}
		return oldValues;
	}

	private float[] resetColumn(float[] data, int width, int height, int inputChannels, int examples, int c,
			float[] oldValues) {
		int index = 0;
		for (int i = 0; i < inputChannels; i++) {
			for (int h = 0; h < height; h++) {
				int startInd = i * height * width * examples + h * width * examples + c * examples;
				for (int e = 0; e < examples; e++) {
					data[startInd + e] = oldValues[index];
					index++;
				}
			}
		}
		return oldValues;
	}

	private float[] zeroRow(float[] data, int width, int height, int inputChannels, int examples, int r) {
		float[] oldValues = new float[inputChannels * width * examples];
		int index = 0;
		for (int i = 0; i < inputChannels; i++) {
			for (int w = 0; w < width; w++) {
				int startInd = i * height * width * examples + (r * width) * examples + w * examples;
				for (int e = 0; e < examples; e++) {
					data[startInd + e] = 0;
					oldValues[index] = data[startInd + e];
					index++;
				}
			}
		}
		return oldValues;
	}

	private float[] resetRow(float[] data, int width, int height, int inputChannels, int examples, int r,
			float[] oldValues) {
		int index = 0;
		for (int i = 0; i < inputChannels; i++) {
			for (int w = 0; w < width; w++) {
				int startInd = i * height * width * examples + (r * width) * examples + w * examples;
				for (int e = 0; e < examples; e++) {
					data[startInd + e] = oldValues[index];
					index++;
				}
			}
		}
		return oldValues;
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		Matrix kernelMatrix = getReversedKernel(axonsContext.getMatrixFactory(), leftNeurons, rightNeurons,
				weights.getConnectionWeights(), config.getPaddingWidth(), config.getPaddingHeight(),
				config.getStrideWidth(), config.getStrideHeight());
		Matrix biasMatrix = weights.getRightToLeftBiases();

		ImageNeuronsActivation convolution = performConvolution(rightNeuronsActivation, axonsContext, kernelMatrix,
				biasMatrix, rightNeurons, leftNeurons);

		reformatRightToLeftInput(axonsContext.getMatrixFactory(), rightNeuronsActivation);

		rightNeuronsActivation.setImmutable(true);

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, convolution);
	}

	public NeuronsActivation reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {

		if (input.isImmutable()) {
			throw new UnsupportedOperationException();
		} else {
			input.reshape(rightNeurons.getDepth(),
					rightNeurons.getWidth() * rightNeurons.getHeight() * input.getExampleCount());
			return input;
		}
	}

	private static Matrix getReversedKernel(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Matrix kernel, int paddingWidth, int paddingHeight, int strideWidth, int strideHeight) {
		float[] data = kernel.getRowByRowArray();
		float[] reversedData = new float[data.length];
		int inputWidthWithPadding = leftNeurons.getWidth() + paddingWidth * 2;
		int inputHeightWithPadding = leftNeurons.getHeight() + paddingHeight * 2;
		int filterWidth = inputWidthWithPadding + (1 - rightNeurons.getWidth()) * (strideWidth);
		int filterHeight = inputHeightWithPadding + (1 - rightNeurons.getHeight()) * (strideHeight);
		// System.out.println(leftNeurons.getDepth() + ":" + leftNeurons.getWidth() +
		// ":" + leftNeurons.getHeight() + ":" + leftNeurons.getWidth());
		// System.out.println(rightNeurons.getDepth() + ":" + rightNeurons.getWidth() +
		// ":" + rightNeurons.getHeight() + ":" + rightNeurons.getWidth());

		for (int o = 0; o < rightNeurons.getDepth(); o++) {
			for (int i = 0; i < leftNeurons.getDepth(); i++) {
				int start1 = o * leftNeurons.getDepth() * filterWidth * filterHeight + i * filterWidth * filterHeight;
				int start2 = i * rightNeurons.getDepth() * filterWidth * filterHeight + o * filterWidth * filterHeight;

				int end1 = start1 + filterHeight * filterWidth - 1;
				int end2 = start2 + filterHeight * filterWidth - 1;

				for (int h = 0; h < filterHeight; h++) {
					for (int w = 0; w < filterWidth; w++) {
						int offset = h * filterWidth + w;
						// System.out.println((start1 + offset) + ":" + (end1 - offset));
						// System.out.println((start2 + offset) + ":" + (end2- offset));

						reversedData[start1 + offset] = data[end2 - offset];
					}
				}
			}
		}
		return matrixFactory.createMatrixFromRowsByRowsArray(kernel.getRows(), kernel.getColumns(), reversedData);
	}

	@Override
	public ConvolutionalAxons dup() {
		// TODO config.dup()
		return new LowMemorySamePaddingConvolutionalAxonsImpl(leftNeurons, rightNeurons, weights, config);
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return !axonsContext.isWithFreezeOut();
	}

	@Override
	public AxonWeights getDetachedAxonWeights() {
		return weights.dup();
	}

	@Override
	public Axons3DConfig getConfig() {
		return config;
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.equals(format);
	}
}
