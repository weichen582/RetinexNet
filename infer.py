from model import lowlight_enhance, load_images
import tensorflow as tf
import os


def lowlight_test(input_file, lowlight_enhance):
    test_low_data_name = [input_file]
    test_low_data = []
    test_high_data = []
    for i in range(1):
        print('fileload', test_low_data_name[i])
        test_low_im = load_images(test_low_data_name[i])
        print('fileload return', test_low_im)
        test_low_data.append(test_low_im)

    lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, save_dir='test_results', decom_flag=0)


def main(input_file, use_gpu=False):
    print('called main')
    if use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            lowlight_test(input_file, model)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            lowlight_test(input_file, model)


#if __name__ == '__main__':
#    main(input_file, use_gpu=False)

        #tf.app.run()