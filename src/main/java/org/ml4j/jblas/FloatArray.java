package org.ml4j.jblas;

public class FloatArray {

	private float[] data;
	private int offset;	
	
	public FloatArray(float[] data, int offset) {
		this.data = data;
		this.offset = offset;
	}

	public float[] getData() {
		return data;
	}

	public int getOffset() {
		return offset;
	}
}
