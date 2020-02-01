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

/**
 * JBlasRowMajorMatrixFactory which creates optimised versions of JBlasRowMajorMatrix
 * (  JBlasRowMajorMatrixOptimised instances).s
 * 
 * @author Michael Lavelle
 */
public class JBlasRowMajorMatrixFactoryOptimised extends JBlasRowMajorMatrixFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	protected Matrix createJBlasMatrix(FloatMatrix matrix, boolean immutable) {
		return new JBlasRowMajorMatrixOptimised(this, floatMatrixFactory, floatArrayFactory, matrix, immutable);
	}
}
