// InfluenceLayers.ts
// Moduł analizy wielowarstwowego systemu wpływu dla MIGI
// Autor: Patryk Sobierański - Meta-Geniusz®

export type ArchetypeType = 
  | "Hero" | "Mother" | "Shadow" | "Trickster" | "Sage" | "Ruler" 
  | "Innocent" | "Explorer" | "Creator" | "Caregiver" | "Rebel" | "Lover";

export type InfluenceLayer = {
  name: string;
  intensity: number; // 0.0 to 1.0
  dominantArchetype: ArchetypeType;
  emotionalSignature: string;
  mediaChannels: string[];
};

export type NarrativePattern = {
  archetype: ArchetypeType;
  frequency: number;
  emotionalImpact: number;
  culturalOrigin: string;
  modernManifestations: string[];
};

export class InfluenceLayers {
  private activeArchetypes: Map<ArchetypeType, number> = new Map();
  private narrativePatterns: NarrativePattern[] = [];
  private influenceLayers: InfluenceLayer[] = [];

  constructor() {
    this.initializeArchetypes();
    this.loadDefaultNarrativePatterns();
  }

  // Wykrywa aktywne archetypy w tekście/mediach
  detectActiveArchetypes(content: string): ArchetypeType[] {
    const detectedArchetypes: ArchetypeType[] = [];
    
    // Wzorce językowe dla każdego archetypu
    const archetypePatterns = {
      "Hero": ["bohater", "walka", "misja", "zwycięstwo", "odwaga"],
      "Shadow": ["ukryty", "tajny", "mroczny", "kontrola", "spisek"],
      "Trickster": ["chaos", "zmiana", "humor", "przekręt", "iluzja"],
      "Mother": ["opieka", "troska", "rodzina", "bezpieczeństwo", "dom"],
      "Sage": ["mądrość", "wiedza", "prawda", "zrozumienie", "nauka"],
      "Ruler": ["władza", "porządek", "hierarchia", "kontrola", "system"]
    };

    Object.entries(archetypePatterns).forEach(([archetype, patterns]) => {
      if (patterns.some(pattern => content.toLowerCase().includes(pattern))) {
        detectedArchetypes.push(archetype as ArchetypeType);
      }
    });

    return detectedArchetypes;
  }

  // Analizuje warstwy wpływu w systemie
  analyzeInfluenceLayers(mediaData: any[]): InfluenceLayer[] {
    const layers: InfluenceLayer[] = [
      {
        name: "Narrative Layer",
        intensity: this.calculateNarrativeIntensity(mediaData),
        dominantArchetype: this.getDominantArchetype(),
        emotionalSignature: "fear-based",
        mediaChannels: ["social_media", "news", "entertainment"]
      },
      {
        name: "Emotional Layer", 
        intensity: this.calculateEmotionalIntensity(mediaData),
        dominantArchetype: "Shadow",
        emotionalSignature: "anxiety-inducing",
        mediaChannels: ["algorithms", "notifications", "feeds"]
      },
      {
        name: "Technological Layer",
        intensity: 0.8,
        dominantArchetype: "Ruler",
        emotionalSignature: "control-oriented",
        mediaChannels: ["AI_systems", "platforms", "data_collection"]
      },
      {
        name: "Social Layer",
        intensity: 0.7,
        dominantArchetype: "Innocent",
        emotionalSignature: "conformity-pressure",
        mediaChannels: ["peer_groups", "influencers", "trends"]
      }
    ];

    this.influenceLayers = layers;
    return layers;
  }

  // Identyfikuje wzorce narracyjne
  identifyNarrativePatterns(content: string[]): NarrativePattern[] {
    const patterns: NarrativePattern[] = [];
    
    content.forEach(text => {
      const archetypes = this.detectActiveArchetypes(text);
      archetypes.forEach(archetype => {
        const existingPattern = patterns.find(p => p.archetype === archetype);
        if (existingPattern) {
          existingPattern.frequency++;
        } else {
          patterns.push({
            archetype,
            frequency: 1,
            emotionalImpact: Math.random() * 0.8 + 0.2,
            culturalOrigin: "modern_media",
            modernManifestations: ["social_media", "entertainment", "advertising"]
          });
        }
      });
    });

    return patterns.sort((a, b) => b.frequency - a.frequency);
  }

  // Generuje raport wpływu
  generateInfluenceReport(): any {
    return {
      timestamp: new Date().toISOString(),
      dominantArchetypes: this.getTopArchetypes(3),
      influenceLayers: this.influenceLayers,
      narrativePatterns: this.narrativePatterns,
      riskAssessment: this.assessInfluenceRisk(),
      recommendations: this.generateRecommendations()
    };
  }

  // Ocenia ryzyko manipulacji
  assessInfluenceRisk(): string {
    const shadowIntensity = this.activeArchetypes.get("Shadow") || 0;
    const rulerIntensity = this.activeArchetypes.get("Ruler") || 0;
    
    if (shadowIntensity > 0.7 && rulerIntensity > 0.6) {
      return "HIGH - Detected patterns of psychological manipulation";
    } else if (shadowIntensity > 0.5 || rulerIntensity > 0.5) {
      return "MEDIUM - Some manipulative patterns detected";
    } else {
      return "LOW - Balanced archetypal distribution";
    }
  }

  // Generuje rekomendacje przeciwdziałania
  generateRecommendations(): string[] {
    const recommendations = [];
    const dominantArchetype = this.getDominantArchetype();
    
    switch (dominantArchetype) {
      case "Shadow":
        recommendations.push("Activate Hero archetype through empowering narratives");
        recommendations.push("Increase transparency and light-based content");
        break;
      case "Ruler":
        recommendations.push("Promote Rebel and Trickster archetypes for balance");
        recommendations.push("Encourage individual thinking and creativity");
        break;
      case "Hero":
        recommendations.push("Balance with Sage archetype for wisdom");
        recommendations.push("Add Caregiver elements for emotional stability");
        break;
      default:
        recommendations.push("Maintain current archetypal balance");
    }
    
    return recommendations;
  }

  // Pomocnicze metody
  private initializeArchetypes(): void {
    const archetypes: ArchetypeType[] = [
      "Hero", "Mother", "Shadow", "Trickster", "Sage", "Ruler",
      "Innocent", "Explorer", "Creator", "Caregiver", "Rebel", "Lover"
    ];
    
    archetypes.forEach(archetype => {
      this.activeArchetypes.set(archetype, Math.random() * 0.5);
    });
  }

  private loadDefaultNarrativePatterns(): void {
    this.narrativePatterns = [
      {
        archetype: "Hero",
        frequency: 15,
        emotionalImpact: 0.8,
        culturalOrigin: "universal",
        modernManifestations: ["superhero_movies", "startup_culture", "self_help"]
      },
      {
        archetype: "Shadow", 
        frequency: 12,
        emotionalImpact: 0.9,
        culturalOrigin: "universal",
        modernManifestations: ["conspiracy_theories", "dark_media", "fear_based_news"]
      }
    ];
  }

  private calculateNarrativeIntensity(data: any[]): number {
    return Math.random() * 0.8 + 0.2; // Placeholder
  }

  private calculateEmotionalIntensity(data: any[]): number {
    return Math.random() * 0.8 + 0.2; // Placeholder
  }

  private getDominantArchetype(): ArchetypeType {
    let maxIntensity = 0;
    let dominant: ArchetypeType = "Hero";
    
    this.activeArchetypes.forEach((intensity, archetype) => {
      if (intensity > maxIntensity) {
        maxIntensity = intensity;
        dominant = archetype;
      }
    });
    
    return dominant;
  }

  private getTopArchetypes(count: number): ArchetypeType[] {
    return Array.from(this.activeArchetypes.entries())
      .sort(([,a], [,b]) => b - a)
      .slice(0, count)
      .map(([archetype]) => archetype);
  }
}

// Eksport dla integracji z MIGI
export default InfluenceLayers;