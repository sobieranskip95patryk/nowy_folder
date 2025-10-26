// PinkPlayEvo with SWR Integration Example
// Przykład kompletnej integracji MŚWR z PinkPlayEvo

const express = require('express');
const cors = require('cors');
const { PinkPlaySWR, createSWRMiddleware } = require('./swrModule');

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// SWR Middleware
const swrMiddleware = createSWRMiddleware();
app.use('/api/generate', swrMiddleware);

// Initialize SWR
const swr = new PinkPlaySWR();

// Routes
app.post('/api/generate', async (req, res) => {
  try {
    const { story, user_id, generation_options = {} } = req.body;
    
    if (!story) {
      return res.status(400).json({ 
        error: 'Story is required',
        code: 'MISSING_STORY'
      });
    }

    // SWR już przetworzył story przez middleware
    const swrEnhanced = req.swr_enhanced;
    
    // Symulacja generacji video (w rzeczywistości tutaj byłby pipeline AI)
    const videoGeneration = await simulateVideoGeneration(
      swrEnhanced.enhanced_prompt,
      swrEnhanced.technical_params,
      swrEnhanced.style_suggestions
    );

    // Response z kompletną metadatą SWR
    res.json({
      success: true,
      original_story: story,
      enhanced_story: swrEnhanced.enhanced_prompt,
      generation_ready: swrEnhanced.generation_ready,
      swr_analysis: {
        quality_score: swrEnhanced.quality_score,
        sentiment: swrEnhanced.swr_metadata.sentiment,
        residuals_found: swrEnhanced.swr_metadata.residuals_found,
        cognitive_coherence: swrEnhanced.swr_metadata.cognitive_coherence
      },
      style_suggestions: swrEnhanced.style_suggestions,
      technical_params: swrEnhanced.technical_params,
      video_result: videoGeneration,
      processing_metadata: {
        timestamp: new Date().toISOString(),
        user_id: user_id || 'anonymous'
      }
    });

  } catch (error) {
    console.error('❌ Generation error:', error);
    res.status(500).json({
      error: 'Generation failed',
      message: error.message,
      code: 'GENERATION_ERROR'
    });
  }
});

// SWR Analytics endpoint
app.get('/api/swr/analytics', async (req, res) => {
  try {
    const analytics = swr.getAnalytics();
    res.json({
      success: true,
      analytics: analytics,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      error: 'Analytics failed',
      message: error.message
    });
  }
});

// Batch processing endpoint
app.post('/api/generate/batch', async (req, res) => {
  try {
    const { stories, user_id } = req.body;
    
    if (!Array.isArray(stories) || stories.length === 0) {
      return res.status(400).json({
        error: 'Stories array is required',
        code: 'MISSING_STORIES'
      });
    }

    if (stories.length > 10) {
      return res.status(400).json({
        error: 'Maximum 10 stories per batch',
        code: 'BATCH_TOO_LARGE'
      });
    }

    const batchResult = await swr.batchProcessStories(stories, user_id);
    
    res.json({
      success: true,
      batch_result: batchResult,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('❌ Batch processing error:', error);
    res.status(500).json({
      error: 'Batch processing failed',
      message: error.message,
      code: 'BATCH_ERROR'
    });
  }
});

// Health check z SWR status
app.get('/health', async (req, res) => {
  try {
    const swrStatus = swr.getAnalytics();
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      swr_initialized: swrStatus.initialized,
      total_processed: swrStatus.totalProcessed,
      average_quality: swrStatus.average_quality.toFixed(3)
    });
  } catch (error) {
    res.status(500).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});

// Symulacja generacji video (placeholder)
async function simulateVideoGeneration(prompt, technicalParams, styleParams) {
  // W rzeczywistości tutaj byłby prawdziwy pipeline AI
  await new Promise(resolve => setTimeout(resolve, Math.random() * 2000 + 1000));
  
  return {
    video_id: `video_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    status: 'generated',
    duration: 7,
    resolution: '1024x1024',
    format: 'mp4',
    generation_time_ms: Math.floor(Math.random() * 5000 + 3000),
    technical_params_used: technicalParams,
    style_params_used: styleParams,
    url: `https://pinkplayevo.com/videos/video_${Date.now()}.mp4`
  };
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

// Start server
async function startServer() {
  try {
    // Initialize SWR
    await swr.initialize();
    
    app.listen(port, () => {
      console.log(`🚀 PinkPlayEvo SWR Server running on port ${port}`);
      console.log(`🧠 SWR Module initialized and ready`);
      console.log(`📡 Available endpoints:`);
      console.log(`   POST /api/generate - Generate video with SWR enhancement`);
      console.log(`   POST /api/generate/batch - Batch process stories`);
      console.log(`   GET /api/swr/analytics - SWR analytics`);
      console.log(`   GET /health - Health check`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  startServer();
}

module.exports = { app, swr };