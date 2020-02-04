package org.ml4j.jblas;

import org.jblas.FloatMatrix;

public class FloatArrayMatrix {

	private FloatArray floatArray;
	private int rows;
	private int columns;
	
	public float[] getRowByRowArray() {
		return floatArray.getData();
	}
	
	public FloatArrayMatrix(FloatMatrix floatMatrix) {
		this.floatArray = new FloatArray(floatMatrix.data, 0);
		this.rows = floatMatrix.getRows();
		this.columns = floatMatrix.getColumns();
	}
	
	public FloatArrayMatrix(FloatArray floatArray, int rows, int columns) {
		this.floatArray = floatArray;
		this.rows = rows;
		this.columns = columns;
	}
	public FloatArray getFloatArray() {
		return floatArray;
	}
	public int getRows() {
		return rows;
	}
	public int getColumns() {
		return columns;
	}
	
	public int getOffset() {
		return floatArray.getOffset();
	}
}
