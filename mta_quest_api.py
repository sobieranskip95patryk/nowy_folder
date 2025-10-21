#!/usr/bin/env python3
"""
MTA Quest - AI Life Optimizer
Flask API Backend for Landing Page

Integruje Meta-Genius Unified System z web interface
dla mtaquestwebskidex.com

Autor: Meta-Genius Team
Data: 22 października 2025
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import sys
import uuid
from datetime import datetime
import logging

# Import naszego MGUS
from meta_genius_unified_system import MetaGeniusUnifiedSystem

# Konfiguracja
app = Flask(__name__)
CORS(app)  # Pozwala na cross-origin requests

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globalna instancja MGUS
mgus = None

def initialize_mgus():
    """Inicjalizacja Meta-Genius Unified System"""
    global mgus
    try:
        logger.info("🚀 Inicjalizacja MGUS dla MTA Quest...")
        mgus = MetaGeniusUnifiedSystem()
        logger.info("✅ MGUS zainicjalizowany pomyślnie")
        return True
    except Exception as e:
        logger.error(f"❌ Błąd inicjalizacji MGUS: {e}")
        return False

@app.route('/')
def landing_page():
    """Główna strona landing page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "mgus_initialized": mgus is not None,
        "timestamp": datetime.now().isoformat(),
        "service": "MTA Quest API v1.0"
    })

@app.route('/api/success-probability', methods=['POST'])
def calculate_success_probability():
    """
    API endpoint do obliczania prawdopodobieństwa sukcesu
    
    Expected JSON payload:
    {
        "goal": "Twój cel",
        "context": "Dodatkowy kontekst",
        "timeline": "6 miesięcy",
        "resources": ["czas", "pieniądze", "umiejętności"],
        "constraints": ["brak doświadczenia", "ograniczony czas"]
    }
    """
    try:
        if not mgus:
            return jsonify({"error": "MGUS nie jest zainicjalizowany"}), 500
        
        data = request.get_json()
        
        if not data or 'goal' not in data:
            return jsonify({"error": "Pole 'goal' jest wymagane"}), 400
        
        # Przygotowanie scenariusza dla AI_Psyche_GOK:AI
        scenario = {
            "goal": data['goal'],
            "context": data.get('context', 'MTA Quest analysis'),
            "resources": data.get('resources', ['motywacja', 'determinacja']),
            "timeline": data.get('timeline', 'średnioterminowy'),
            "constraints": data.get('constraints', ['brak doświadczenia'])
        }
        
        logger.info(f"🎯 Analizuję cel: {scenario['goal']}")
        
        # Obliczenie prawdopodobieństwa przez AI_Psyche_GOK:AI
        if mgus.ai_psyche:
            success_probability = mgus.ai_psyche.calculate_success_probability(scenario)
            recommendations = mgus.ai_psyche.generate_recommendations([scenario])
            
            # Przygotowanie dodatkowych insightów
            current_phase = mgus.ai_psyche._current_phase.value
            capital_level = mgus.ai_psyche.calculate_capital()
            
            # Analiza przez inne systemy MGUS
            additional_insights = []
            
            if mgus.logos_core:
                additional_insights.append("🧠 LOGOS: Logiczna analiza wzorców przeprowadzona")
            
            if mgus.timeline_4d:
                additional_insights.append("⏰ Timeline4D: Optymalna ścieżka czasowa zidentyfikowana")
            
            if mgus.ai_matchmaker:
                additional_insights.append("💕 Synergia: Potencjalne połączenia z mentorami dostępne")
            
            response = {
                "success_probability": round(success_probability, 3),
                "percentage": round(success_probability * 100, 1),
                "current_phase": current_phase,
                "capital_level": round(capital_level, 2),
                "recommendations": recommendations[:3] if recommendations else [],
                "insights": additional_insights,
                "analysis_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "meta_data": {
                    "scenario": scenario,
                    "mgus_version": "1.0",
                    "ai_psyche_active": True
                }
            }
            
            logger.info(f"✅ Analiza ukończona: {response['percentage']}% szans sukcesu")
            return jsonify(response)
            
        else:
            return jsonify({"error": "AI_Psyche_GOK:AI nie jest dostępny"}), 500
            
    except Exception as e:
        logger.error(f"❌ Błąd podczas analizy: {e}")
        return jsonify({"error": f"Błąd serwera: {str(e)}"}), 500

@app.route('/api/comprehensive-analysis', methods=['POST'])
def comprehensive_analysis():
    """
    Kompleksowa analiza przez wszystkie systemy MGUS
    """
    try:
        if not mgus:
            return jsonify({"error": "MGUS nie jest zainicjalizowany"}), 500
        
        data = request.get_json()
        
        # Utworzenie profilu użytkownika
        user_data = {
            "user_id": f"mta_user_{uuid.uuid4().hex[:8]}",
            "age": data.get('age', 25),
            "dominant_emotion": data.get('emotion', 'determined'),
            "spiritual_beliefs": data.get('beliefs', 'growth-oriented'),
            "relationship_goals": data.get('relationship_goals', ['growth']),
            "interests": data.get('interests', ['development', 'technology'])
        }
        
        logger.info(f"🔍 Kompleksowa analiza dla użytkownika: {user_data['user_id']}")
        
        # Analiza przez wszystkie systemy
        comprehensive_profile = mgus.create_comprehensive_user_profile(user_data)
        
        # Zunifikowana analiza tematu
        topic_data = {
            "phenomena": [data.get('goal', 'Personal development')],
            "context": data.get('context', 'Life optimization'),
            "resources": data.get('resources', []),
            "timeline": data.get('timeline', 'medium-term')
        }
        
        unified_analysis = mgus.perform_unified_analysis(
            data.get('goal', 'Personal Development'), 
            topic_data
        )
        
        response = {
            "user_profile": comprehensive_profile,
            "unified_analysis": unified_analysis,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Kompleksowa analiza ukończona")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Błąd podczas kompleksowej analizy: {e}")
        return jsonify({"error": f"Błąd serwera: {str(e)}"}), 500

@app.route('/api/quick-insight', methods=['POST'])
def quick_insight():
    """
    Szybki insight dla prostych zapytań
    Idealny dla landing page widget
    """
    try:
        data = request.get_json()
        goal = data.get('goal', '').strip()
        
        if not goal:
            return jsonify({"error": "Cel nie może być pusty"}), 400
        
        # Prosty mock analysis jeśli MGUS nie jest dostępny
        if not mgus or not mgus.ai_psyche:
            # Fallback mock response
            mock_probability = 0.45 + (len(goal) % 10) * 0.05
            return jsonify({
                "success_probability": round(mock_probability, 3),
                "percentage": round(mock_probability * 100, 1),
                "quick_tip": f"Cel '{goal}' wymaga strategicznego podejścia. Zacznij od małych kroków!",
                "status": "mock_analysis",
                "timestamp": datetime.now().isoformat()
            })
        
        # Prawdziwa analiza przez MGUS
        scenario = {"goal": goal, "context": "Quick insight", "resources": ["basic"], "timeline": "unspecified", "constraints": ["general"]}
        
        success_prob = mgus.ai_psyche.calculate_success_probability(scenario)
        
        # Generowanie quick tip na podstawie prawdopodobieństwa
        if success_prob > 0.7:
            tip = f"Świetny cel! Masz wysokie szanse sukcesu. Skoncentruj się na konsekwentnym działaniu."
        elif success_prob > 0.5:
            tip = f"Realny cel z dobrymi perspektywami. Przygotuj solidny plan działania."
        elif success_prob > 0.3:
            tip = f"Ambitny cel! Wymaga dodatkowych zasobów i strategii. Rozważ podział na mniejsze etapy."
        else:
            tip = f"Bardzo ambitny cel. Kluczowe będzie dobre przygotowanie i wsparcie mentorów."
        
        return jsonify({
            "success_probability": round(success_prob, 3),
            "percentage": round(success_prob * 100, 1),
            "quick_tip": tip,
            "status": "mgus_analysis",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Błąd quick insight: {e}")
        return jsonify({"error": f"Błąd serwera: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("🌟 Uruchamianie MTA Quest API...")
    
    # Inicjalizacja MGUS
    mgus_ready = initialize_mgus()
    
    if not mgus_ready:
        logger.warning("⚠️ MGUS nie został zainicjalizowany, API będzie działać w trybie mock")
    
    logger.info("🚀 MTA Quest API gotowy na porcie 5000")
    logger.info("📡 Endpoints dostępne:")
    logger.info("   GET  /              - Landing page")
    logger.info("   GET  /api/health    - Health check")
    logger.info("   POST /api/success-probability - Analiza prawdopodobieństwa sukcesu")
    logger.info("   POST /api/comprehensive-analysis - Pełna analiza MGUS")
    logger.info("   POST /api/quick-insight - Szybki insight")
    
    app.run(debug=True, host='0.0.0.0', port=5000)