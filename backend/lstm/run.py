import ncnn
import numpy as np
import onnx
import onnxruntime as rt

if __name__ == '__main__':
    sess = rt.InferenceSession("best_sim.onnx")
    x_val = np.array([[[0.1 for i in range(6)] for j in range(128)]]).astype(np.float32)
    print(x_val.shape)

#result = sess.run(["123"], {"x": x_val})
    result = sess.run(["output"], {"input": x_val})
    print(result)


    with ncnn.Net() as net:
        ret = net.load_param("best.param")
        print(ret)
        net.load_model("best.bin")
        in_mat = ncnn.Mat((128, 6, 1))
        in_mat.fill(0.1)
        

        with net.create_extractor() as ex:
            ex.input("input", in_mat)
#ret, out_mat = ex.extract("123")
            ret, out_mat = ex.extract("output")
            print(out_mat)
