
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
package org.ml4j.nn.factories;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.DefaultConvolutionalAxonsImpl;
import org.ml4j.nn.axons.DefaultOneByOneConvolutionalAxonsImpl;
import org.ml4j.nn.axons.LowMemorySamePaddingConvolutionalAxonsImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Custom implementation of DefaultAxonsFactoryImpl optimised for memory.
 * 
 * @author Michael Lavelle
 */
public class LowMemoryOptimisedDefaultAxonsFactoryImpl extends DefaultAxonsFactoryImpl {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public LowMemoryOptimisedDefaultAxonsFactoryImpl(MatrixFactory matrixFactory) {
		super(matrixFactory);
	}

	@Override
	public ConvolutionalAxons createConvolutionalAxons(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config, WeightsMatrix connectionWeights, BiasMatrix biases) {
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());
		if (DefaultOneByOneConvolutionalAxonsImpl.isEligible(leftNeurons, rightNeurons, config)) {
			return new DefaultOneByOneConvolutionalAxonsImpl(this, leftNeurons, rightNeurons,  config, connectionWeights, biases);
		} else if (filterHeight == filterWidth && config.getStrideHeight() == 1 && config.getStrideWidth() == 1 && leftNeurons.getWidth() == rightNeurons.getWidth() && leftNeurons.getHeight() == rightNeurons.getHeight()){
			return new LowMemorySamePaddingConvolutionalAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config, connectionWeights, biases);
		}
		else {
			return new DefaultConvolutionalAxonsImpl(this, leftNeurons, rightNeurons, config, connectionWeights, biases);
		}
	}
}
