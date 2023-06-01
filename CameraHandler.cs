using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Threading.Tasks;
using Xamarin.Forms;
using Xamarin.Essentials;
using TensorFlowLite;
using ObjectDetection;

namespace ObjectDetection
{
    public partial class MainPage : ContentPage
    {
        private ModelInterpreter modelInterpreter;
        private CameraHandler cameraHandler;
        private List<BoxView> overlayBoxes;

        public MainPage()
        {
            InitializeComponent();

            // Initialize the model interpreter with the model path and labels
            string modelPath = "model.tflite";
            List<string> labels = LoadLabels("..\\labels.txt");
            modelInterpreter = new ModelInterpreter(modelPath, labels);

            // Initialize the camera handler
            cameraHandler = new CameraHandler();
            cameraHandler.FrameArrived += CameraHandler_FrameArrived;

            overlayBoxes = new List<BoxView>();
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();

            // Request camera permission
            var status = await Permissions.RequestAsync<Permissions.Camera>();
            if (status != PermissionStatus.Granted)
            {
                // Camera permission not granted, handle accordingly
                return;
            }

            await cameraHandler.StartPreviewAsync();
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();

            // Stop the camera preview and cleanup resources
            cameraHandler.StopPreview();
        }

        private async void CameraHandler_FrameArrived(object sender, byte[] imageBytes)
        {
            // Run object detection on the received frame
            List<ModelInterpreter.DetectionResult> results = await modelInterpreter.DetectAsync(imageBytes);

            // Clear previous detection overlays
            ClearOverlayBoxes();

            // Update the UI with the detection results
            Device.BeginInvokeOnMainThread(() =>
            {
                foreach (var result in results)
                {
                    // Draw detection overlay on the frame
                    DrawOverlayBox(result.X1, result.Y1, result.X2, result.Y2);
                }
            });
        }

        private void ClearOverlayBoxes()
        {
            foreach (var box in overlayBoxes)
            {
                canvas.Children.Remove(box);
            }
            overlayBoxes.Clear();
        }

        private void DrawOverlayBox(float x1, float y1, float x2, float y2)
        {
            var box = new BoxView
            {
                Color = Color.Red,
                Opacity = 0.5,
                AnchorX = 0,
                AnchorY = 0
            };
            box.SetValue(AbsoluteLayout.LayoutBoundsProperty, new Rectangle(x1, y1, x2 - x1, y2 - y1));
            box.SetValue(AbsoluteLayout.LayoutFlagsProperty, AbsoluteLayoutFlags.All);
            overlayBoxes.Add(box);
            canvas.Children.Add(box);
        }

        private List<string> LoadLabels(string labelsPath)
        {
            List<string> labels = new List<string>();

            if (File.Exists(labelsPath))
            {
                using (StreamReader sr = new StreamReader(labelsPath))
                {
                    string line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        labels.Add(line);
                    }
                }
            }

            return labels;
        }
    }
}
