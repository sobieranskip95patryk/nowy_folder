// swrModule.js - Node.js wrapper dla PinkPlayEvo SWR Integration
// Moduł Świadomego Wnioskowania Resztkowego dla PinkPlayEvo

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

class PinkPlaySWR {
  constructor() {
    this.pythonScript = path.join(__dirname, 'core', 'pinkplay_swr_integration.py');
    this.isInitialized = false;
    this.processingQueue = [];
    this.analytics = {
      totalProcessed: 0,
      averageQuality: 0,
      totalResiduals: 0
    };
  }

  async initialize() {
    try {
      // Sprawdź czy Python script istnieje
      await fs.access(this.pythonScript);
      this.isInitialized = true;
      console.log('🧠 SWR Module initialized successfully');
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize SWR Module:', error.message);
      return false;
    }
  }

  async processStory(story, userId = null, options = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      
      // Przygotuj dane dla Python script
      const inputData = {
        story: story,
        user_id: userId,
        options: options,
        timestamp: new Date().toISOString()
      };

      // Spawn Python process
      const pythonProcess = spawn('python', ['-c', `
import sys
import json
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.append(r'${path.dirname(this.pythonScript).replace(/\\/g, '/')}')
from core.pinkplay_swr_integration import create_pinkplay_swr

# Odczytaj input z stdin
input_data = json.loads(sys.stdin.read())
swr = create_pinkplay_swr()

# Przetwórz fabułę
result = swr.process_story_for_pinkplay(
    input_data['story'], 
    input_data.get('user_id')
)

# Zwróć wynik jako JSON (bez emoji dla Windows compatibility)
result_clean = {k: str(v).encode('ascii', 'ignore').decode('ascii') if isinstance(v, str) else v for k, v in result.items()}
print(json.dumps(result_clean, ensure_ascii=True))
      `], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        const processingTime = Date.now() - startTime;
        
        if (code !== 0) {
          console.error('❌ SWR Python process error:', errorOutput);
          reject(new Error(`SWR processing failed: ${errorOutput}`));
          return;
        }

        try {
          const result = JSON.parse(output);
          
          // Dodaj metadata Node.js
          result.processing_metadata = {
            ...result.processing_metadata,
            node_processing_time_ms: processingTime,
            processed_at: new Date().toISOString()
          };

          // Aktualizuj analitykę
          this._updateAnalytics(result);

          resolve(result);
        } catch (parseError) {
          console.error('❌ Failed to parse SWR result:', parseError);
          reject(new Error('Invalid SWR response format'));
        }
      });

      // Wyślij dane do Python process
      pythonProcess.stdin.write(JSON.stringify(inputData));
      pythonProcess.stdin.end();

      // Timeout protection
      setTimeout(() => {
        pythonProcess.kill();
        reject(new Error('SWR processing timeout'));
      }, 30000); // 30s timeout
    });
  }

  _updateAnalytics(result) {
    this.analytics.totalProcessed++;
    this.analytics.averageQuality = (
      (this.analytics.averageQuality * (this.analytics.totalProcessed - 1) + 
       result.quality_score) / this.analytics.totalProcessed
    );
    this.analytics.totalResiduals += result.residuals.length;
  }

  async batchProcessStories(stories, userId = null) {
    const results = [];
    const batchStartTime = Date.now();

    console.log(`🔄 Processing batch of ${stories.length} stories...`);

    for (let i = 0; i < stories.length; i++) {
      try {
        const result = await this.processStory(stories[i], userId, { batchIndex: i });
        results.push({
          index: i,
          success: true,
          result: result
        });
        console.log(`✅ Story ${i + 1}/${stories.length} processed (Quality: ${result.quality_score.toFixed(3)})`);
      } catch (error) {
        results.push({
          index: i,
          success: false,
          error: error.message
        });
        console.error(`❌ Story ${i + 1}/${stories.length} failed:`, error.message);
      }
    }

    const batchTime = Date.now() - batchStartTime;
    console.log(`🎯 Batch processing completed in ${batchTime}ms`);

    return {
      total_stories: stories.length,
      successful: results.filter(r => r.success).length,
      failed: results.filter(r => !r.success).length,
      batch_processing_time_ms: batchTime,
      results: results
    };
  }

  getAnalytics() {
    return {
      ...this.analytics,
      average_residuals_per_story: this.analytics.totalProcessed > 0 
        ? this.analytics.totalResiduals / this.analytics.totalProcessed 
        : 0,
      initialized: this.isInitialized
    };
  }

  // Integration z PinkPlayEvo pipeline
  async enhancePromptForGeneration(story, generationOptions = {}) {
    try {
      const swrResult = await this.processStory(story, null, {
        generation_context: true,
        style: generationOptions.style || 'standard',
        duration: generationOptions.duration || 7
      });

      // Zwróć enhanced prompt wraz z metadatą dla generacji
      return {
        original_prompt: story,
        enhanced_prompt: swrResult.enhanced_story,
        generation_ready: swrResult.ready_for_generation,
        quality_score: swrResult.quality_score,
        style_suggestions: this._generateStyleSuggestions(swrResult),
        technical_params: this._generateTechnicalParams(swrResult),
        swr_metadata: {
          residuals_found: swrResult.residuals.length,
          sentiment: swrResult.sentiment_analysis.dominant_sentiment,
          cognitive_coherence: swrResult.mswr_insights.cognitive_coherence
        }
      };
    } catch (error) {
      console.error('❌ SWR enhancement failed:', error);
      // Fallback - zwróć oryginalny prompt
      return {
        original_prompt: story,
        enhanced_prompt: story,
        generation_ready: true,
        quality_score: 0.7,
        style_suggestions: {},
        technical_params: {},
        swr_metadata: { error: error.message }
      };
    }
  }

  _generateStyleSuggestions(swrResult) {
    const suggestions = {};
    const sentiment = swrResult.sentiment_analysis.dominant_sentiment;

    // Style mapping based na sentyment
    if (sentiment === 'positive') {
      suggestions.color_palette = 'warm_bright';
      suggestions.movement_style = 'dynamic_uplifting';
    } else if (sentiment === 'negative') {
      suggestions.color_palette = 'cool_dramatic';
      suggestions.movement_style = 'contemplative_slow';
    } else if (sentiment === 'intense') {
      suggestions.color_palette = 'high_contrast';
      suggestions.movement_style = 'dramatic_powerful';
    }

    // Dodatkowe sugestie based na residuals
    for (const residual of swrResult.residuals) {
      if (residual.type === 'low_emotional_density') {
        suggestions.emotional_boost = 'increase_visual_drama';
      } else if (residual.type === 'lack_of_action') {
        suggestions.movement_enhancement = 'add_dynamic_elements';
      }
    }

    return suggestions;
  }

  _generateTechnicalParams(swrResult) {
    const params = {};
    
    // Parametry techniczne based na quality score
    if (swrResult.quality_score > 0.8) {
      params.inference_steps = 50; // Higher quality
      params.guidance_scale = 7.5;
    } else {
      params.inference_steps = 30; // Standard quality
      params.guidance_scale = 7.0;
    }

    // Adjustment based na cognitive coherence
    if (swrResult.mswr_insights.cognitive_coherence > 0.5) {
      params.prompt_strength = 0.8; // Strong adherence
    } else {
      params.prompt_strength = 0.6; // More creative freedom
    }

    return params;
  }
}

// Express middleware dla PinkPlayEvo
function createSWRMiddleware() {
  const swr = new PinkPlaySWR();

  return async (req, res, next) => {
    if (req.body && req.body.story) {
      try {
        console.log('🧠 SWR Processing story...');
        const enhanced = await swr.enhancePromptForGeneration(
          req.body.story, 
          req.body.generation_options || {}
        );

        // Dodaj enhanced prompt do request
        req.swr_enhanced = enhanced;
        req.body.enhanced_story = enhanced.enhanced_prompt;

        // Dodaj SWR metadata do response
        res.locals.swr_metadata = enhanced.swr_metadata;

        console.log(`✅ SWR Enhancement completed (Quality: ${enhanced.quality_score.toFixed(3)})`);
      } catch (error) {
        console.error('❌ SWR Middleware error:', error);
        // Continue bez enhancement
      }
    }
    next();
  };
}

module.exports = {
  PinkPlaySWR,
  createSWRMiddleware
};

// Test jeśli uruchomiony bezpośrednio
if (require.main === module) {
  async function test() {
    console.log('🧪 Testing PinkPlayEvo SWR Module...');
    
    const swr = new PinkPlaySWR();
    await swr.initialize();

    const testStories = [
      "Młoda kobieta tańczy w deszczu, czując wolność.",
      "Bohater walczy z demonami wewnętrznymi.",
      "Kot śpi na parapecie."
    ];

    for (const story of testStories) {
      try {
        const result = await swr.enhancePromptForGeneration(story);
        console.log(`📊 Story: "${story.substring(0, 30)}..."`);
        console.log(`📈 Quality: ${result.quality_score.toFixed(3)}`);
        console.log(`✨ Enhanced: "${result.enhanced_prompt.substring(0, 50)}..."`);
        console.log('---');
      } catch (error) {
        console.error('❌ Test failed:', error.message);
      }
    }

    console.log('📈 Analytics:', swr.getAnalytics());
    console.log('🎯 SWR Module test completed!');
  }

  test().catch(console.error);
}