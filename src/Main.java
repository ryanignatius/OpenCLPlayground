import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLBuildException;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.util.IOUtils;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;

/**
 * Main
 * {Default file description, change this}
 * Copyright (c) 2018 GDP Labs. All rights reserved.
 *
 * @author ryan-i-hadiwijaya
 * @since Mar 14, 2018.
 */
public class Main {

    public static void main(String[] args) {
        CLContext context = JavaCL.createBestContext();

        CLQueue queue = context.createDefaultQueue();
        int n = 1024;

        /*
        ByteOrder byteOrder = context.getByteOrder();

        Pointer<Float>
            aPtr = allocateFloats(n).order(byteOrder),
            bPtr = allocateFloats(n).order(byteOrder);

        for (int i = 0; i < n; i++) {
            aPtr.set(i, (float)i);
            //aPtr.set(i, (float)cos(i));
            bPtr.set(i, (float)(i*i));
            //bPtr.set(i, (float)sin(i));
        }
        */

        // Create OpenCL input buffers (using the native memory pointers aPtr and bPtr) :
        CLBuffer<Float>
            a = context.createFloatBuffer(CLMem.Usage.Input, n),
            b = context.createFloatBuffer(CLMem.Usage.Input, n);

        // Create an OpenCL output buffer :
        CLBuffer<Float> out = context.createFloatBuffer(CLMem.Usage.Output, n);

        String programText = "";
        try {
            File file = new File("src/kernel.cl");
            programText = IOUtils.readText(file);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        CLProgram program = context.createProgram(programText);
        try {
            program.build();
            System.out.println("program build successful");
        } catch (CLBuildException ex) {
            ex.printStackTrace();
        }

        // Get and call the kernel :
        CLKernel fillValuesKernel = program.createKernel("fill_in_values");
        fillValuesKernel.setArgs(a, b, n);
        int[] globalSizes2 = new int[] { n };
        CLEvent fillEvt = fillValuesKernel.enqueueNDRange(queue, globalSizes2);

        FloatBuffer outPtr2 = out.read(queue, fillEvt); // blocks until fill_in_values finished

        CLKernel addFloatsKernel = program.createKernel("add_floats");
        addFloatsKernel.setArgs(a, b, out, n);
        int[] globalSizes = new int[] { n };
        CLEvent addEvt = addFloatsKernel.enqueueNDRange(queue, globalSizes);

        FloatBuffer outPtr = out.read(queue, addEvt); // blocks until add_floats finished
        System.out.println(outPtr);

        // Print the first 10 output values :
        for (int i = 0; i < 10 && i < n; i++) {
            System.out.println("out[" + i + "] = " + outPtr.get(i));
        }
    }
}
