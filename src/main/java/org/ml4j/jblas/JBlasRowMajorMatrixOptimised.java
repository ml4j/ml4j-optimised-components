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
package org.ml4j.jblas;

import org.jblas.FloatMatrix;
import org.ml4j.Matrix;
import org.ml4j.floatarray.FloatArrayFactory;
import org.ml4j.floatmatrix.FloatMatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrix;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;

import com.github.fommil.netlib.BLAS;

/**
 * JBlasRowMajorMatrix subclass with optimisations.
 * 
 * @author Michael Lavelle
 *
 */
public class JBlasRowMajorMatrixOptimised extends JBlasRowMajorMatrix {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public JBlasRowMajorMatrixOptimised(JBlasRowMajorMatrixFactory jblasRowMajorMatrixFactory,
			FloatMatrixFactory floatMatrixFactory, FloatArrayFactory floatArrayFactory, FloatMatrix matrix,
			boolean immutable) {
		super(jblasRowMajorMatrixFactory, floatMatrixFactory, floatArrayFactory, matrix, immutable);
	}

	/**
	 * Optimised mmul method which uses com.github.fommil.netlib.BLAS sgemm.
	 */
	@Override
	public Matrix mmul(Matrix other) {
		FloatMatrix otherMatrix = createJBlasFloatMatrix(other);
		FloatMatrix thisMatrix = this.matrix;
		float[] cdata = floatArrayFactory.createFloatArray(other.getColumns() * getRows());
		FloatMatrix c = floatMatrixFactory.create(other.getColumns(), getRows(), cdata);
		float alpha = 1f;
		float beta =1f;
		int resultOffset = 0;
		FloatMatrix result = gemm(alpha, otherMatrix, thisMatrix, beta, c, resultOffset);
		return createJBlasMatrix(result, false);
	}
	
	/**
	 * Optimised mmul method which uses com.github.fommil.netlib.BLAS sgemm.
	 */
	@Override
	public Matrix mmul(Matrix other, Matrix target) {
		FloatMatrix otherMatrix = createJBlasFloatMatrix(other);
		FloatMatrix targetMatrix = createJBlasFloatMatrix(target);
		FloatMatrix thisMatrix = this.matrix;
		float alpha = 1f;
		float beta =1f;
		int resultOffset = 0;
		FloatMatrix result = gemm(alpha, otherMatrix, thisMatrix, beta, targetMatrix, resultOffset);
		return createJBlasMatrix(result, false);
	}

	/**
	 * Compute c <- alpha * a*b + beta * c (general matrix matrix
	 * multiplication)
	 * 
	 * @param alpha
	 * @param a
	 * @param b
	 * @param beta
	 * @param c
	 * @param cOffset
	 * @return
	 */
	public static FloatMatrix gemm(float alpha, FloatMatrix a,
			FloatMatrix b, float beta, FloatMatrix c, int cOffset) {
		BLAS.getInstance().sgemm("N", "N", c.rows, c.columns, a.columns, alpha, a.data, 0,
				a.rows, b.data, 0, b.rows, beta, c.data, cOffset, c.rows);
		return c;
	}
}
