package org.ml4j.nn.axons;

import java.util.Arrays;

import org.junit.Before;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.Axons3DTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public class LowMemorySamePaddingConvolutionalAxonsImplTest extends Axons3DTestBase<ConvolutionalAxons> {

	@Before
	@Override
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		super.setUp();
		Mockito.when(mockAxonsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockLeftToRightInputActivation = createNeuronsActivation(784 * 3, 32);
	}

	@Override
	protected ConvolutionalAxons createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		
		Mockito.when(leftNeurons.getNeuronCountExcludingBias()).thenReturn(784 * 3);
		Mockito.when(leftNeurons.getDepth()).thenReturn(3);
		Mockito.when(leftNeurons.getWidth()).thenReturn(28);
		Mockito.when(leftNeurons.getHeight()).thenReturn(28);
		Mockito.when(rightNeurons.getNeuronCountExcludingBias()).thenReturn(28 * 28 * 2);
		Mockito.when(rightNeurons.getDepth()).thenReturn(2);
		Mockito.when(rightNeurons.getWidth()).thenReturn(28);
		Mockito.when(rightNeurons.getHeight()).thenReturn(28);
		
		config.withPaddingHeight(1);
		config.withPaddingWidth(1);
		config.withFilterHeight(3);
		config.withFilterWidth(3);
		
		return new LowMemorySamePaddingConvolutionalAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config, 
				new WeightsMatrixImpl(null, new WeightsFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH), 
						Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH), 
						WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)), null);
	}

	@Override
	public void testPushLeftToRight() {

		super.testPushLeftToRight();
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(new Neurons(featureCount, false),
				matrixFactory.createMatrix(featureCount, exampleCount),
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT);
	}

	@Override
	protected int getExpectedReformattedInputColumns() {
		return 32 * 28 * 28;
	}

	@Override
	protected int getExpectedReformattedInputRows() {
		return 3 * 3 * 3;
	}

	@Override
	protected boolean expectPostDropoutInputToBeSet() {
		return true;
	}

}
