using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Xamarin.Forms.PlatformConfiguration;

namespace Object_Detection_App
{
    public class TensorflowObjectDetector : IObjectDetector
    {
        const int FloatSize = 4;
        const int PixelSize = FloatSize * 3;

        private float[][] _OutputBoxes;
        public Java.Lang.Object OutputBoxes
        {
            get => Java.Lang.Object.FromArray(_OutputBoxes);
            set => _OutputBoxes = value.ToArray<float[]>();
        }

        private float[] _OutputScores;
        public Java.Lang.Object OutputScores
        {
            get => Java.Lang.Object.FromArray(_OutputScores);
            set => _OutputScores = value.ToArray<float>();
        }

        private long[] _OutputClasses;
        public Java.Lang.Object OutputClasses
        {
            get => Java.Lang.Object.FromArray(_OutputClasses);
            set => _OutputClasses = value.ToArray<long>();
        }

        public Interpreter Interpreter { get; }

        public TensorflowObjectDetector()
        {
            var mappedByteBuffer = GetModelAsMappedByteBuffer();
            Interpreter = new Interpreter(mappedByteBuffer);
        }

        public ImagePrediction Detect(byte[] image)
        {
            var tensor = Interpreter.GetInputTensor(0);
            var shape = tensor.Shape();

            var width = shape[1];
            var height = shape[2];

            using var imageByteBuffer = GetPhotoAsByteBuffer(image, width, height);

            var numDetections = Interpreter.GetOutputTensor(0).NumElements();

            _OutputBoxes = CreateJaggedArray<float>(numDetections, 4);
            _OutputScores = new float[numDetections];
            _OutputClasses = new long[numDetections];

            Java.Lang.Object[] inputArray = { imageByteBuffer };

            var outputMap = new Dictionary<Java.Lang.Integer, Java.Lang.Object>();
            outputMap.Add(new Java.Lang.Integer(0), OutputBoxes);
            outputMap.Add(new Java.Lang.Integer(1), OutputScores);
            outputMap.Add(new Java.Lang.Integer(2), OutputClasses);

            Interpreter.RunForMultipleInputsOutputs(inputArray, outputMap);

            var imagePrediction = new ImagePrediction(predictions: new List<PredictionModel>());

            for (var i = 0; i < _OutputScores.Length; i++)
            {
                var probability = _OutputScores[i];
                var label = _OutputClasses[i].ToString();
                imagePrediction.Predictions.Add(new PredictionModel(probability, label));
            }

            return imagePrediction;
        }

        private MappedByteBuffer GetModelAsMappedByteBuffer()
        {
            var assetDescriptor = Android.App.Application.Context.Assets.OpenFd("model.tflite");

            using var inputStream = new FileInputStream(assetDescriptor.FileDescriptor);

            var mappedByteBuffer = inputStream.Channel.Map(
                FileChannel.MapMode.ReadOnly,
                assetDescriptor.StartOffset,
                assetDescriptor.DeclaredLength
            );

            return mappedByteBuffer;
        }

        private ByteBuffer GetPhotoAsByteBuffer(byte[] image, int width, int height)
        {
            var bitmap = BitmapFactory.DecodeByteArray(image, 0, image.Length);
            var resizedBitmap = Bitmap.CreateScaledBitmap(bitmap, width, height, true);

            var modelInputSize = height * width * PixelSize;
            var byteBuffer = ByteBuffer.AllocateDirect(modelInputSize);
            byteBuffer.Order(ByteOrder.NativeOrder());

            var pixels = new int[width * height];
            resizedBitmap.GetPixels(
                pixels,
                0,
                resizedBitmap.Width,
                0,
                0,
                resizedBitmap.Width,
                resizedBitmap.Height
            );

            var pixel = 0;

            for (var i = 0; i < width; i++)
            {
                for (var j = 0; j < height; j++)
                {
                    var pixelVal = pixels[pixel++];

                    byteBuffer.PutFloat(pixelVal >> 16 & 0xFF);
                    byteBuffer.PutFloat(pixelVal >> 8 & 0xFF);
                    byteBuffer.PutFloat(pixelVal & 0xFF);
                }
            }

            bitmap.Recycle();

            return byteBuffer;
        }

        private static T[][] CreateJaggedArray<T>(int lay1, int lay2)
        {
            var arr = new T[lay1][];

            for (int i = 0; i < lay1; i++)
            {
                arr[i] = new T[lay2];
            }

            return arr;
        }
    }
}
