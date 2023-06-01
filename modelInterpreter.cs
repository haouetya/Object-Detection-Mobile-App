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
        private readonly float threshold;

        private readonly Interpreter interpreter;
        private readonly int inputImageSize;
        private readonly int outputBoxCount;
        private readonly int outputClassCount;
        private readonly int outputBoxSize;

        public ModelInterpreter(string modelPath, float threshold = 0.5f)
        {
            this.modelPath = modelPath;
            this.threshold = threshold;

            interpreter = new Interpreter(modelPath);
            inputImageSize = 300;
            outputBoxCount = 0; // Update with the correct index of the "locations" output tensor
            outputClassCount = 0; // Update with the correct index of the "classes" output tensor
            outputBoxSize = 0; // Update with the correct index of the "scores" output tensor
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
                { outputBoxCount, outputLocations },
                { outputClassCount, outputClasses },
                { outputBoxSize, outputScores }
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
