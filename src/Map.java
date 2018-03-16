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
 * Map
 * {Default file description, change this}
 * Copyright (c) 2018 GDP Labs. All rights reserved.
 *
 * @author ryan-i-hadiwijaya
 * @since Mar 16, 2018.
 */
public class Map {

    public void run() {
        int n = 1<<26;
        long startCPU = System.currentTimeMillis();
        runCPU(n);
        long stopCPU = System.currentTimeMillis();
        long startCL = System.currentTimeMillis();
        runCL(n);
        long stopCL = System.currentTimeMillis();
        System.out.println("CPU time: " + (stopCPU - startCPU) + " milliseconds");
        System.out.println("CL time: " + (stopCL - startCL) + " milliseconds");
    }

    public void runCPU(int n) {
        /*
        Float a[] = new Float[n];
        Float b[] = new Float[n];
        Float out[] = new Float[n];
        for (int i=0; i<n; i++) {
            a[i] = ((float)i);
            b[i] = ((float)(i*9));
            out[i] = a[i] + b[i];
        }
        // Print the first 10 output values :
        for (int i = 0; i < 10 && i < n; i++) {
            System.out.println("out[" + i + "] = " + out[i]);
        }

        // Print the last 10 output values :
        for (int i = n - 10; i < n; i++) {
            System.out.println("out[" + i + "] = " + out[i]/100000);
        }
        */
        int a[] = new int[n];
        int b[] = new int[n];
        int out[] = new int[n];
        for (int i=0; i<n; i++) {
            a[i] = i;
            b[i] = (i*9);
            out[i] = a[i] + b[i];
        }
        // Print the first 10 output values :
        for (int i = 0; i < 10 && i < n; i++) {
            System.out.println("out[" + i + "] = " + out[i]);
        }

        // Print the last 10 output values :
        for (int i = n - 10; i < n; i++) {
            System.out.println("out[" + i + "] = " + out[i]);
        }
    }

    public void runCL(int n) {
        CLContext context = JavaCL.createBestContext();

        CLQueue queue = context.createDefaultQueue();

        // Create OpenCL input buffers (using the native memory pointers aPtr and bPtr) :
        CLBuffer<Float>
            a = context.createFloatBuffer(CLMem.Usage.Input, n),
            b = context.createFloatBuffer(CLMem.Usage.Input, n);

        // Create an OpenCL output buffer :
        CLBuffer<Float> out = context.createFloatBuffer(CLMem.Usage.Output, n);

        String programText = "";
        try {
            File file = new File("src/map.cl");
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

        CLKernel addFloatsKernel = program.createKernel("add_floats");
        addFloatsKernel.setArgs(a, b, out, n);
        int[] globalSizes = new int[] { n };
        CLEvent addEvt = addFloatsKernel.enqueueNDRange(queue, globalSizes, fillEvt);

        FloatBuffer outPtr = out.read(queue, addEvt); // blocks until add_floats finished

        // Print the first 10 output values :
        for (int i = 0; i < 10 && i < n; i++) {
            System.out.println("out[" + i + "] = " + outPtr.get(i));
        }

        // Print the last 10 output values :
        for (int i = n - 10; i < n; i++) {
            System.out.println("out[" + i + "] = " + outPtr.get(i)/100000);
        }
    }
}
