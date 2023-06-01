using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xamarin.Forms;
using TensorFlowLite;

namespace ObjectDetection
{
    public class ModelInterpreter
    {
        private readonly string modelPath;
        private readonly List<string> labels;
        private readonly float threshold;

        private readonly Interpreter interpreter;
        private readonly int inputImageSize;
        private readonly int outputBoxCount;
        private readonly int outputClassCount;
        private readonly int outputBoxSize;

        public ModelInterpreter(string modelPath, List<string> labels, float threshold = 0.5f)
        {
            this.modelPath = modelPath;
            this.labels = labels;
            this.threshold = threshold;

            interpreter = new Interpreter(modelPath);
            var inputShape = interpreter.GetInputTensorShape(0);
            inputImageSize = inputShape[1];
            var outputShape = interpreter.GetOutputTensorShape(0);
            outputBoxCount = outputShape[1];
            outputClassCount = outputShape[2];
            outputBoxSize = outputShape[3];
        }

        public async Task<List<DetectionResult>> DetectAsync(byte[] imageBytes)
        {
            var image = ImageUtils.DecodeJpeg(imageBytes, inputImageSize, inputImageSize);
            var outputLocations = new float[1, outputBoxCount, outputBoxSize];
            var outputClasses = new float[1, outputBoxCount, outputClassCount];
            var outputScores = new float[1, outputBoxCount];
            var inputs = new List<object> { image };
            var outputs = new Dictionary<int, object>
            {
                { 0, outputLocations },
                { 1, outputClasses },
                { 2, outputScores }
            };
            interpreter.RunForMultipleInputsOutputs(inputs, outputs);

            var results = new List<DetectionResult>();
            for (var i = 0; i < outputBoxCount; i++)
            {
                var classId = outputClasses[0, i, 0];
                var score = outputScores[0, i];
                if (score < threshold) continue;

                var label = labels[(int)classId];
                var box = outputLocations[0, i, new[] { 1, 0, 3, 2 }]; // reorder boxes to [x1, y1, x2, y2]

                var x1 = box[0];
                var y1 = box[1];
                var x2 = box[2];
                var y2 = box[3];

                results.Add(new DetectionResult(label, score, x1, y1, x2, y2));
            }

            return results;
        }

        public class DetectionResult
        {
            public string Label { get; }
            public float Score { get; }
            public float X1 { get; }
            public float Y1 { get; }
            public float X2 { get; }
            public float Y2 { get; }

            public DetectionResult(string label, float score, float x1, float y1, float x2, float y2)
            {
                Label = label;
                Score = score;
                X1 = x1;
                Y1 = y1;
                X2 = x2;
                Y2 = y2;
            }
        }
    }
}
