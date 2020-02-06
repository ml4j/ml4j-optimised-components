package org.ml4j.nn.axons;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.neurons.Neurons3D;

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
			return new DefaultOneByOneConvolutionalAxonsImpl(this, leftNeurons, rightNeurons, config, connectionWeights, biases);
		} else if (filterHeight == filterWidth && config.getStrideHeight() == 1 && config.getStrideWidth() == 1 && leftNeurons.getWidth() == rightNeurons.getWidth() && leftNeurons.getHeight() == rightNeurons.getHeight()){
			return new LowMemorySamePaddingConvolutionalAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config, connectionWeights, biases, null);
		}
		else {
			return new DefaultConvolutionalAxonsImpl(this, leftNeurons, rightNeurons, config, connectionWeights, biases);
		}
	}
}
