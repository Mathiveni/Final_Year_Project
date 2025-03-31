import React, { useState } from 'react';
import styles from "../styles/searchpage.module.css";
import uploadIcon from "../images/icons/medscan.png";
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import CircularProgress from '@mui/material/CircularProgress';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogActions';
import DialogActions from '@mui/material/DialogActions';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const SearchPage = () => {
  const [loading, setLoading] = useState(false);
  const [resultDialogOpen, setResultDialogOpen] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleSearchImage = () => {
    const imgView = document.getElementById('img-view');
    const backgroundImage = imgView.style.backgroundImage;

    if ((!backgroundImage || backgroundImage === 'none')) {
      alert('Please upload an image before searching.');
      return;
    }

    const formData = new FormData();
    const fileInput = document.getElementById('input-file');
    formData.append('image', fileInput.files[0]);

    setLoading(true);
    setErrorMessage(null);

    fetch('http://localhost:8080/predict', {
      method: 'POST',
      body: formData
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        return response.json();
      })
      .then(data => {
        setLoading(false);
        setPredictionResult(data);
        setResultDialogOpen(true);
      })
      .catch(error => {
        setLoading(false);
        console.error('Error predicting disease:', error);
        setErrorMessage('Failed to process image. Please try again.');
        alert('Failed to process image. Please try again.');
      });
  };

  const uploadImage = (event) => {
    const file = event.target.files[0];
    if (!file) {
      console.error("No file selected.");
      return;
    }
    try {
      const imgLink = URL.createObjectURL(file);
      document.getElementById('img-view').style.backgroundImage = `url(${imgLink})`;
      document.getElementById('img-view').textContent = '';
      document.getElementById('img-view').style.border = 'none';
    }
    catch (error) {
      console.error("Failed to create object URL:", error);
    }
  };

  const resetImage = () => {
    document.getElementById('input-file').value = '';
    document.getElementById('img-view').style.backgroundImage = 'none';
    document.getElementById('img-view').textContent = 'Upload Your Image Here';
    document.getElementById('img-view').style.border = '2px dashed #ccc';
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      const fileInput = document.getElementById('input-file');
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInput.files = dataTransfer.files;

      const imgLink = URL.createObjectURL(file);
      document.getElementById('img-view').style.backgroundImage = `url(${imgLink})`;
      document.getElementById('img-view').textContent = '';
      document.getElementById('img-view').style.border = 'none';
    }
  };

  const handleCloseDialog = () => {
    setResultDialogOpen(false);
  };

  const prepareChartData = () => {
    if (!predictionResult || !predictionResult.all_probabilities) return [];

    return Object.entries(predictionResult.all_probabilities).map(([category, probability]) => ({
      category,
      probability: probability.toFixed(1)
    }));
  };

  // Helper function to get severity color
  const getSeverityColor = (severityScore) => {
    if (severityScore < 25) return 'bg-green-500';
    if (severityScore < 40) return 'bg-yellow-300';
    if (severityScore < 60) return 'bg-yellow-500';
    if (severityScore < 80) return 'bg-orange-500';
    return 'bg-red-500';
  };

  return (
    <div className={styles.searchpage}>
      <div className={styles.upload}>
        <label htmlFor="input-file" className={styles.dropArea} onDrop={handleDrop} onDragOver={handleDragOver}>
          <input type="file" accept="image/*" id="input-file" onChange={uploadImage} hidden />
          <div id="img-view" className={styles.imgView}>
            <img src={uploadIcon} alt="upload icon" className={styles.icon} />
            <p className={styles.uploadtext}>Upload Your Image Here</p>
          </div>
        </label>
        <div className={styles.bin}>
          <IconButton aria-label="delete" size="large" onClick={resetImage}>
            <DeleteIcon fontSize="inherit" />
          </IconButton>
        </div>
      </div>

      <div className={styles.searchButtonSet}>
        <Button
          variant="contained"
          size="large"
          onClick={handleSearchImage}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Analyze OCT Image'}
        </Button>
      </div>

      {/* Result Dialog */}
      <Dialog
        open={resultDialogOpen}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Eye Disease Analysis Result</DialogTitle>
        <DialogContent>
          {predictionResult && (
            <div className={styles.resultContainer}>
              <div className={styles.resultSummary}>
                <h2>Prediction: {predictionResult.prediction}</h2>
                <h3>Confidence: {predictionResult.confidence.toFixed(2)}%</h3>
              </div>

              {/* Severity Assessment */}
              <div className={styles.severityAssessment}>
                <h3>Severity Assessment</h3>
                <div className="flex items-center space-x-4">
                  <div
                    className={`h-6 w-full ${getSeverityColor(predictionResult.severity_score)}`}
                    style={{ width: `${predictionResult.severity_score}%`, transition: 'width 0.5s ease-in-out' }}
                  />
                  <span>
                    Severity Level: {predictionResult.severity_level.toUpperCase()}
                    ({predictionResult.severity_score.toFixed(2)}%)
                  </span>
                </div>

                {/* Severity Component Details */}
                <div className="mt-4">
                  <h4>Severity Components:</h4>
                  {Object.entries(predictionResult.severity_components).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span>{key.replace(/_/g, ' ')}</span>
                      <span>{value.toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Severity Explanation */}
              <div className={styles.severityExplanation}>
                <h3>Severity Explanation</h3>
                <p>{predictionResult.severity_explanation}</p>
              </div>

              {/* Image Comparison Section */}
              <div className={styles.imageComparison}>
                <div className={styles.originalImage}>
                  <h3>Original Image</h3>
                  <img
                    src={`data:image/png;base64,${predictionResult.original_image}`}
                    alt="Original OCT Scan"
                    style={{ maxWidth: '100%', maxHeight: '300px' }}
                  />
                </div>
                <div className={styles.originalImage}>
                  <h3>Grad-CAM Heatmap</h3>
                  <img
                    src={`data:image/png;base64,${predictionResult.heatmap_image}`}
                    alt="Grad-CAM Heatmap"
                    style={{ maxWidth: '100%', maxHeight: '300px' }}
                  />
                </div>
              </div>

              {/* Heatmap Interpretation */}
              <div className={styles.heatmapInterpretation}>
                <h3>Heatmap Interpretation</h3>
                <p>{predictionResult.heatmap_interpretation}</p>
              </div>

              {/* Probability Chart */}
              <div className={styles.chartContainer} style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={prepareChartData(predictionResult.all_probabilities)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value) => [`${value}%`, 'Probability']} />
                    <Bar dataKey="probability" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Disease Explanation */}
              <div className={styles.diseaseInfo}>
                <h3>Detailed Explanation:</h3>
                <p>{predictionResult.explanation}</p>
              </div>
            </div>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} color="primary">Close</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default SearchPage;
