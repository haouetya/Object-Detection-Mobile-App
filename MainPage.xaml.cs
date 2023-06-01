using Plugin.Media;
using Plugin.Media.Abstractions;
using Xamarin.Forms;
using Xamarin.GoogleMLKit.ObjectDetection;
using System.Linq;
using static Xamarin.Google.MLKit.Vision.Objects.ObjectDetectorOptionsBase;

namespace ObjectDetection
{
    public partial class MainPage : ContentPage
    {
        private ObjectDetector objectDetector;

        public MainPage()
        {
            InitializeComponent();
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();

            // Request camera and storage permissions
            await CrossMedia.Current.Initialize();

            // Load the object detection model
            objectDetector = ObjectDetection.GetObjectDetector(new CustomObjectDetectorOptions.Builder()
                .SetDetectorMode(DetectorMode.Stream)
                .SetClassificationConfidenceThreshold(0.5f)
                .Build());

            // Start capturing from the camera
            CameraView.StartScanning();
            CameraView.FrameReceived += OnFrameReceived;
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();

            // Stop capturing from the camera
            CameraView.StopScanning();
            CameraView.FrameReceived -= OnFrameReceived;

            // Release the object detection model
            objectDetector.Dispose();
        }

        private async void OnFrameReceived(object sender, Xamarin.Forms.CameraView.FrameEventArgs e)
        {
            // Resize the image to match the input size of the object detection model
            var image = e.Frame.Resize(new Size(640, 640));

            // Detect objects in the image
            var results = await objectDetector.ProcessImageAsync(image);

            // Show the label of the first detected object on the screen
            var detectedObject = results.FirstOrDefault();
            if (detectedObject != null)
            {
                Label.Text = detectedObject.ClassificationCategory.DisplayName;
                AbsoluteLayout.SetLayoutBounds(BoxView, detectedObject.BoundingBox);
                BoxView.Color = Color.FromHex("#80FF0000");
            }
            else
            {
                Label.Text = "";
                BoxView.Color = Color.Transparent;
            }
        }

        float threshold = 0.5f; // Specify the threshold value

        List<ModelInterpreter.DetectionResult> filteredResults = results
            .Where(result => result.Score >= threshold)
            .ToList();
    }
}
