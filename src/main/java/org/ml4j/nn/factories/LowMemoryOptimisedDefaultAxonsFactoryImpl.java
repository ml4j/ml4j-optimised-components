
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
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.DefaultConvolutionalAxonsImpl;
import org.ml4j.nn.axons.DefaultOneByOneConvolutionalAxonsImpl;
import org.ml4j.nn.axons.LowMemorySamePaddingConvolutionalAxonsImpl;
import org.ml4j.nn.axons.WeightsMatrix;

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
	public ConvolutionalAxons createConvolutionalAxons(
			Axons3DConfig config, WeightsMatrix connectionWeights, BiasVector biases) {
	
		if (DefaultOneByOneConvolutionalAxonsImpl.isEligible(config)) {
			return new DefaultOneByOneConvolutionalAxonsImpl(this, config, connectionWeights, biases);
		} else if (config.getFilterHeight() == config.getFilterWidth() && config.getStrideHeight() == 1 && config.getStrideWidth() == 1 && config.getLeftNeurons().getWidth() == config.getRightNeurons().getWidth() && config.getLeftNeurons().getHeight() == config.getRightNeurons().getHeight()){
			return new LowMemorySamePaddingConvolutionalAxonsImpl(matrixFactory, config, connectionWeights, biases);
		}
		else {
			return new DefaultConvolutionalAxonsImpl(this, config, connectionWeights, biases);
		}
	}
}
