
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
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;

/**
 * Prototype experimental implementation of k2r-aa convolutional method from 
 * "Low-memory GEMM-based convolution algorithms for deep neural networks"  ( https://arxiv.org/pdf/1709.03395.pdf )
 * 
 * @author Michael Lavelle
 *
 */
public class LowMemorySamePaddingConvolutionalAxonsImpl implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;
	private Axons3DConfig config;
	private AxonWeights convolutionalAxonWeights;
	
	public LowMemorySamePaddingConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config, AxonWeights convolutionalAxonWeights) {
		super();
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
		this.config.setFilterWidthAndHeight(leftNeurons, rightNeurons);
		this.convolutionalAxonWeights = convolutionalAxonWeights;
	}

	public LowMemorySamePaddingConvolutionalAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config,
			WeightsMatrix weightsMatrix, BiasMatrix biasMatrix) {
		this(leftNeurons, rightNeurons, config, 
				createInitialAxonWeights(matrixFactory, leftNeurons, rightNeurons, config, weightsMatrix, biasMatrix));
	}
	
	private static AxonWeights createInitialAxonWeights(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config, WeightsMatrix connectionWeights, BiasMatrix leftToRightBiases) {
		
		if (connectionWeights == null) {
			throw new IllegalArgumentException("WeightsMatrix cannot be null");
		}
		
		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultFullyConnectedAxonWeightsInitialiser(
				new Neurons(config.getFilterWidth() * config.getFilterHeight() * leftNeurons.getDepth(), leftNeurons.hasBiasUnit()),
				new Neurons(rightNeurons.getDepth(), rightNeurons.hasBiasUnit()));

		Matrix initialConnectionWeights = connectionWeights.getWeights() == null
				? axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory)
				: connectionWeights.getWeights();
							
		Optional<Matrix> initialLeftToRightBiases = leftToRightBiases == null
				? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(leftToRightBiases.getWeights());
				
		return new LowMemorySamePaddingConvolutionalAxonWeightsImpl(leftNeurons, 
				rightNeurons, config, new WeightsMatrixImpl(initialConnectionWeights,
						connectionWeights.getFormat()), leftNeurons.hasBiasUnit() && initialLeftToRightBiases.isPresent() ? new BiasMatrixImpl(initialLeftToRightBiases.get())
						: null);		
			
	}
	
	

	@Override
	public void adjustAxonWeights(AxonWeightsAdjustment adjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		convolutionalAxonWeights.adjustWeights(adjustment, adjustmentDirection);
	}

	@Override
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
		return rightNeurons;
	}

	public NeuronsActivation reformatLeftToRightInput(MatrixFactory matrixFactory,
			NeuronsActivation leftNeuronsActivation) {

		final NeuronsActivation reformatted;
			ImageNeuronsActivation imageAct = leftNeuronsActivation.asImageNeuronsActivation(leftNeurons, DimensionScope.INPUT);
			reformatted = new NeuronsActivationImpl(
					new Neurons(leftNeurons.getDepth() * config.getFilterWidth() * config.getFilterWidth(), leftNeurons.hasBiasUnit()),
					imageAct.im2ColConv(matrixFactory, config),
					ImageNeuronsActivationFormat.ML4J_IM_TO_COL_CONV_FORMAT);
			if (!imageAct.isImmutable()) {
				imageAct.close();
			}
	
		reformatted.setImmutable(true);

		return reformatted;

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		leftNeuronsActivation.setImmutable(true);

		NeuronsActivation output = convolutionalAxonWeights.applyToLeftToRightInput(leftNeuronsActivation.dup(), axonsContext);
		
		Supplier<NeuronsActivation> reformattedSupplier = () -> reformatLeftToRightInput(axonsContext.getMatrixFactory(),
				leftNeuronsActivation);

		if (!leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		}

		return new AxonsActivationImpl(this, null, reformattedSupplier, output);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		NeuronsActivation output = convolutionalAxonWeights.applyToRightToLeftInput(rightNeuronsActivation, axonsContext);
		reformatRightToLeftInput(axonsContext.getMatrixFactory(), rightNeuronsActivation);
		rightNeuronsActivation.setImmutable(true);
		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, output);
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

	@Override
	public ConvolutionalAxons dup() {
		return new LowMemorySamePaddingConvolutionalAxonsImpl(leftNeurons, rightNeurons, config.dup(), convolutionalAxonWeights.dup());
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return !axonsContext.isWithFreezeOut();
	}

	@Override
	public AxonWeights getDetachedAxonWeights() {
		return convolutionalAxonWeights.dup();
	}

	@Override
	public Axons3DConfig getConfig() {
		return config;
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return convolutionalAxonWeights.optimisedFor();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return convolutionalAxonWeights.isSupported(format) 
				&& NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET
			.equals(format.getFeatureOrientation());
	}

	@Override
	public AxonsType getAxonsType() {
		return AxonsType.getBaseType(AxonsBaseType.CONVOLUTIONAL);
	}
}
