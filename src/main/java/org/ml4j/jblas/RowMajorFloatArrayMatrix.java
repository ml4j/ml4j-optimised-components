package org.ml4j.jblas;

public class RowMajorFloatArrayMatrix {

	private FloatArray floatArray;
	private int rows;
	private int columns;
	
	public float[] getRowByRowArray() {
		return floatArray.getData();
	}
	
	public RowMajorFloatArrayMatrix(FloatArray floatArray, int rows, int columns) {
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
