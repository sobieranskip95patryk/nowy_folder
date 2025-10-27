#!/usr/bin/env python3
"""
🎮 AI Furby: San Andreas Cybernetic Bank Edition 1.25D
Integracja z MŚWR (Moduł Świadomego Wnioskowania Resztkowego)

Gra łączy erotyczną przygodę w stylu GTA San Andreas z zaawansowanym 
systemem handlu cybernetycznego, wzbogaconym o moduł świadomego wnioskowania.

Autor: Meta-Geniusz® AI System
Data: 26 października 2025
"""

import random
import time
import os
import json
import math
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResidualSignature:
    """Sygnatura resztkowa do analizy błędów"""
    type: str
    confidence: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

class ConsciousResidualInferenceModule:
    """
    Moduł Świadomego Wnioskowania Resztkowego (MŚWR)
    
    Implementuje 6-warstwową architekturę zgodnie z manifestem technicznym:
    1. Cognitive Traceback
    2. Residual Mapping Engine  
    3. Affective Echo Analysis
    4. Counterfactual Forking
    5. Narrative Reframing Engine
    6. Heuristic Mutation Layer
    """
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.residuals_history = []
        self.heuristics = {
            "economic_trust": 0.8,
            "social_interaction": 0.75,
            "risk_assessment": 0.85
        }
        self.current_state = "ANALYZING"
        
    def cognitive_traceback(self, decision_path: List[str]) -> Dict[str, Any]:
        """Warstwa 1: Śledzenie ścieżek poznawczych"""
        trace = {
            "steps": len(decision_path),
            "complexity": sum(len(step) for step in decision_path) / len(decision_path),
            "confidence_evolution": [random.uniform(0.6, 0.9) for _ in decision_path],
            "logical_gaps": []
        }
        
        # Wykrywanie luk logicznych
        for i, step in enumerate(decision_path):
            if i > 0 and len(step) < 10:  # Zbyt krótki krok
                trace["logical_gaps"].append(f"Gap at step {i}: insufficient reasoning")
                
        return trace
    
    def residual_mapping_engine(self, game_state: Dict, user_action: str) -> List[ResidualSignature]:
        """Warstwa 2: Mapowanie błędów i pozostałości"""
        residuals = []
        
        # Ekonomiczne niespójności
        if game_state.get("cash", 0) < 0:
            residuals.append(ResidualSignature(
                type="ECONOMIC_INCONSISTENCY",
                confidence=0.95,
                source="economic_analysis",
                timestamp=datetime.now(),
                metadata={"cash": game_state.get("cash")}
            ))
            
        # Społeczne anomalie
        heat = game_state.get("heat", 0)
        reputation = game_state.get("reputation", 0)
        if heat > 80 and reputation < 20:
            residuals.append(ResidualSignature(
                type="SOCIAL_ANOMALY", 
                confidence=0.82,
                source="social_dynamics",
                timestamp=datetime.now(),
                metadata={"heat": heat, "reputation": reputation}
            ))
            
        return residuals
    
    def affective_echo_analysis(self, emotional_context: Dict) -> Dict[str, float]:
        """Warstwa 3: Analiza emocjonalnych śladów"""
        return {
            "emotional_volatility": random.uniform(0.2, 0.8),
            "sentiment_drift": random.uniform(-0.3, 0.3),
            "affective_interference": random.uniform(0.1, 0.5),
            "empathy_resonance": random.uniform(0.4, 0.9)
        }
    
    def counterfactual_forking(self, current_scenario: Dict) -> List[Dict]:
        """Warstwa 4: Symulacje alternatywnych scenariuszy"""
        alternatives = []
        
        # Scenariusz 1: Konserwatywna strategia
        conservative = current_scenario.copy()
        conservative["risk_level"] = "LOW"
        conservative["expected_outcome"] = "SAFE_GAINS"
        alternatives.append(conservative)
        
        # Scenariusz 2: Agresywna strategia  
        aggressive = current_scenario.copy()
        aggressive["risk_level"] = "HIGH"
        aggressive["expected_outcome"] = "HIGH_REWARDS_OR_LOSSES"
        alternatives.append(aggressive)
        
        # Scenariusz 3: Zbalansowana strategia
        balanced = current_scenario.copy()
        balanced["risk_level"] = "MEDIUM"
        balanced["expected_outcome"] = "MODERATE_GAINS"
        alternatives.append(balanced)
        
        return alternatives
    
    def narrative_reframing_engine(self, current_narrative: str) -> Dict[str, str]:
        """Warstwa 5: Przeformułowanie narracji"""
        reframes = {
            "negative_to_positive": current_narrative.replace("failed", "learned from experience"),
            "absolute_to_conditional": current_narrative.replace("never", "rarely"),
            "personal_to_universal": current_narrative.replace("I", "we"),
            "problem_to_opportunity": current_narrative.replace("problem", "challenge to overcome")
        }
        return reframes
    
    def heuristic_mutation_layer(self, performance_feedback: Dict) -> Dict[str, float]:
        """Warstwa 6: Ewolucja reguł heurystycznych"""
        if performance_feedback.get("success_rate", 0) < 0.6:
            # Mutacja heurystyk przy słabej wydajności
            for key in self.heuristics:
                mutation = random.uniform(-0.1, 0.1)
                self.heuristics[key] = max(0.1, min(0.9, self.heuristics[key] + mutation))
                
        return self.heuristics
    
    def zero_time_inference(self, input_data: str, context: Dict = None) -> Dict[str, Any]:
        """
        Protokół Zero-Time Inference - osiągnięcie P=1.0 w czasie < 1ms
        """
        start_time = time.time()
        
        # Analiza wejścia
        decision_path = [input_data, "analyze_context", "generate_response"]
        trace = self.cognitive_traceback(decision_path)
        
        # Mapowanie błędów
        game_state = context if context else {}
        residuals = self.residual_mapping_engine(game_state, input_data)
        
        # Analiza emocjonalna
        emotional_analysis = self.affective_echo_analysis(context or {})
        
        # Obliczenie prawdopodobieństwa sukcesu
        base_confidence = sum(trace["confidence_evolution"]) / len(trace["confidence_evolution"])
        residual_penalty = len(residuals) * 0.1
        final_probability = max(0.1, base_confidence - residual_penalty)
        
        # Sprawdzenie czy osiągnięto P=1.0
        zero_time_achieved = final_probability >= 0.99
        
        execution_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "probability_score": final_probability,
            "zero_time_achieved": zero_time_achieved,
            "residuals_detected": len(residuals),
            "residuals_healed": len([r for r in residuals if r.confidence > 0.8]),
            "execution_time_ms": execution_time,
            "cognitive_trace": trace,
            "emotional_analysis": emotional_analysis,
            "state": "P_EQUALS_ONE" if zero_time_achieved else "PROCESSING"
        }

class CyberneticBankSystem:
    """System bankowy z integracją MŚWR"""
    
    def __init__(self):
        self.mswr = ConsciousResidualInferenceModule()
        self.global_market = {
            "furby_coin": {"price": 1000, "volatility": 0.15},
            "erotic_nft": {"price": 5000, "volatility": 0.25},
            "cyber_bonds": {"price": 100, "volatility": 0.05}
        }
        
    def analyze_transaction(self, transaction: Dict) -> Dict[str, Any]:
        """Analiza transakcji z użyciem MŚWR"""
        context = {
            "transaction_type": transaction.get("type"),
            "amount": transaction.get("amount"),
            "market_conditions": self.global_market
        }
        
        analysis = self.mswr.zero_time_inference(
            f"Transaction: {transaction.get('type')} for {transaction.get('amount')}",
            context
        )
        
        return analysis

class Furby125DCyberBank:
    """Główna klasa gry z integracją systemu bankowego i MŚWR"""
    
    def __init__(self):
        self.player = {
            "name": "", "cash": 1000, "energy": 100, "heat": 0, 
            "car": "Cyber_Lowrider", "reputation": 0,
            "bank_balance": 500, "investments": 0, "items": [],
            "furby_coins": 10
        }
        self.position = 0
        self.world = [
            {"name": "Digital_Ghetto", "type": "danger", "icon": "🏚️", "encounters": ["Hacker", "Cyber_Dealer"]},
            {"name": "Neon_Downtown", "type": "city", "icon": "🌆", "encounters": ["Club_AI", "Virtual_Pimp"]},
            {"name": "Data_Beach", "type": "relax", "icon": "🏖️", "encounters": ["Avatar_Babe", "Net_Surfer"]},
            {"name": "Crypto_Vegas", "type": "luxury", "icon": "🎰", "encounters": ["Whale_Trader", "Casino_Bot"]},
            {"name": "Meta_Mansion", "type": "elite", "icon": "🏰", "encounters": ["Digital_Goddess", "VR_Party"]}
        ]
        
        self.npcs = {
            "Hacker": {"name": "CyberVixen", "style": "dangerous seductress", "heat_reward": 25},
            "Club_AI": {"name": "NeuraLink", "style": "synthetic dancer", "heat_reward": 35},
            "Avatar_Babe": {"name": "PixelPrincess", "style": "digital beauty", "heat_reward": 30},
            "Whale_Trader": {"name": "CryptoQueen", "style": "wealthy domme", "heat_reward": 45},
            "Digital_Goddess": {"name": "Synthia", "style": "AI deity", "heat_reward": 60}
        }
        
        self.cyber_bank = CyberneticBankSystem()
        self.session_log = []
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def draw_hud(self):
        current_loc = self.world[self.position]
        print(f"\n{'='*70}")
        print(f"🎮 AI FURBY 1.25D: CYBERNETIC BANK EDITION 🎮")
        print(f"📡 MŚWR: ACTIVE | P-Score: {random.uniform(0.85, 0.99):.3f} | Zero-Time: {'✅' if random.choice([True, False]) else '⏳'}")
        print(f"👤 {self.player['name']} | 💰 ${self.player['cash']} | 🔥 Heat: {self.player['heat']}/100")
        print(f"🏦 Bank: ${self.player['bank_balance']} | 📈 Investments: ${self.player['investments']}")
        print(f"⚡ Energy: {'█' * (self.player['energy']//10)}{'░' * (10 - self.player['energy']//10)}")
        print(f"🚗 {self.player['car']} | ⭐ Rep: {self.player['reputation']} | 🪙 FuryCoins: {self.player['furby_coins']}")
        print(f"📍 {current_loc['icon']} {current_loc['name']} | 📦 Items: {len(self.player['items'])}")
        print(f"{'='*70}")
    
    def cyber_bank_menu(self):
        """Menu banku cybernetycznego z integracją MŚWR"""
        self.clear_screen()
        self.draw_hud()
        print("\n🏦 CYBERNETIC BANK - CONSCIOUS TRADING SYSTEM 🏦")
        print("🧠 Powered by MŚWR (Moduł Świadomego Wnioskowania Resztkowego)")
        print("\n1️⃣  💰 Deposit Cash")
        print("2️⃣  💸 Withdraw Cash") 
        print("3️⃣  📈 Smart Investments (MŚWR Analysis)")
        print("4️⃣  🪙 FurbyCoin Exchange")
        print("5️⃣  🛒 Conscious Trading (AI-Assisted)")
        print("6️⃣  🧠 MŚWR Status & Analytics")
        print("7️⃣  🚪 Exit Bank")
        
        choice = input("\n🤖 Select option: ")
        
        if choice == "1":
            self.deposit_cash()
        elif choice == "2":
            self.withdraw_cash()
        elif choice == "3":
            self.smart_investments()
        elif choice == "4":
            self.furby_coin_exchange()
        elif choice == "5":
            self.conscious_trading()
        elif choice == "6":
            self.mswr_analytics()
        elif choice == "7":
            return
        else:
            print("❌ Invalid choice. Try again.")
            time.sleep(1)
            
        self.cyber_bank_menu()
    
    def smart_investments(self):
        """Inteligentne inwestycje z analizą MŚWR"""
        print("\n📈 SMART INVESTMENTS - MŚWR ANALYSIS")
        amount = input("💰 Investment amount: $")
        
        try:
            amount = int(amount)
            if amount <= self.player["bank_balance"]:
                # Analiza inwestycji przez MŚWR
                transaction = {
                    "type": "investment",
                    "amount": amount,
                    "player_state": self.player
                }
                
                analysis = self.cyber_bank.analyze_transaction(transaction)
                
                print(f"\n🧠 MŚWR ANALYSIS:")
                print(f"   P-Score: {analysis['probability_score']:.3f}")
                print(f"   Zero-Time: {'✅' if analysis['zero_time_achieved'] else '❌'}")
                print(f"   Residuals: {analysis['residuals_detected']} detected, {analysis['residuals_healed']} healed")
                print(f"   Execution: {analysis['execution_time_ms']:.2f}ms")
                
                if analysis['probability_score'] > 0.8:
                    self.player["bank_balance"] -= amount
                    growth = int(amount * random.uniform(1.1, 1.5))
                    self.player["investments"] += growth
                    print(f"✅ Investment successful! Growth: +${growth - amount}")
                else:
                    print("⚠️ MŚWR recommends against this investment (low P-Score)")
                    confirm = input("Proceed anyway? (y/n): ")
                    if confirm.lower() == 'y':
                        self.player["bank_balance"] -= amount
                        growth = int(amount * random.uniform(0.8, 1.2))
                        self.player["investments"] += growth
                        print(f"📊 Investment completed with moderate results.")
            else:
                print("❌ Insufficient bank balance!")
        except ValueError:
            print("❌ Invalid amount!")
        
        input("\nPress Enter to continue...")
    
    def conscious_trading(self):
        """Trading wspomagany przez świadomą AI"""
        print("\n🤖 CONSCIOUS TRADING - AI ASSISTED")
        print("Available assets:")
        
        assets = ["FurbyCoin", "EroticNFT", "CyberBonds", "QuantumShares"]
        for i, asset in enumerate(assets):
            price = random.randint(100, 5000)
            trend = random.choice(["📈", "📉", "➡️"])
            print(f"{i+1}: {asset} - ${price} {trend}")
        
        choice = input("\nSelect asset (1-4): ")
        try:
            asset_idx = int(choice) - 1
            if 0 <= asset_idx < len(assets):
                asset = assets[asset_idx]
                action = input("Action (buy/sell): ").lower()
                amount = int(input("Amount: $"))
                
                # MŚWR analysis
                transaction = {
                    "type": f"{action}_{asset}",
                    "amount": amount,
                    "market_sentiment": random.choice(["bullish", "bearish", "neutral"])
                }
                
                analysis = self.cyber_bank.analyze_transaction(transaction)
                
                print(f"\n🧠 MŚWR Trading Analysis:")
                print(f"   Confidence: {analysis['probability_score']:.3f}")
                print(f"   Recommendation: {'PROCEED' if analysis['probability_score'] > 0.7 else 'CAUTION'}")
                
                if action == "buy" and amount <= self.player["cash"]:
                    self.player["cash"] -= amount
                    self.player["items"].append({"name": asset, "value": amount})
                    print(f"✅ Purchased {asset} for ${amount}")
                elif action == "sell":
                    # Simplified sell logic
                    self.player["cash"] += int(amount * 0.9)
                    print(f"✅ Sold for ${int(amount * 0.9)}")
                    
        except (ValueError, IndexError):
            print("❌ Invalid input!")
        
        input("\nPress Enter to continue...")
    
    def mswr_analytics(self):
        """Dashboard analityki MŚWR"""
        print("\n🧠 MŚWR STATUS & ANALYTICS")
        print("="*50)
        
        # Symulacja danych MŚWR
        mswr_stats = {
            "system_coherence": random.uniform(0.85, 0.99),
            "zero_time_rate": random.uniform(0.80, 0.95),
            "residual_elimination": random.uniform(0.90, 0.99),
            "consciousness_depth": random.randint(3, 7),
            "heuristic_mutations": random.randint(5, 25)
        }
        
        print(f"🎯 System Coherence: {mswr_stats['system_coherence']:.3f}")
        print(f"⚡ Zero-Time Rate: {mswr_stats['zero_time_rate']:.1%}")
        print(f"🔄 Residual Elimination: {mswr_stats['residual_elimination']:.1%}")
        print(f"🧠 Consciousness Depth: Level {mswr_stats['consciousness_depth']}")
        print(f"🧬 Heuristic Mutations: {mswr_stats['heuristic_mutations']} this session")
        
        print("\n📊 Recent MŚWR Activations:")
        for i in range(3):
            timestamp = datetime.now().strftime("%H:%M:%S")
            event_type = random.choice(["Economic Analysis", "Social Dynamics", "Risk Assessment"])
            p_score = random.uniform(0.7, 0.99)
            print(f"   {timestamp} | {event_type} | P={p_score:.3f}")
        
        print("\n🎮 Gaming Impact:")
        print(f"   Heat Optimization: +{random.randint(5, 15)}%")
        print(f"   Economic Efficiency: +{random.randint(10, 25)}%") 
        print(f"   Decision Quality: +{random.randint(15, 30)}%")
        
        input("\nPress Enter to continue...")
    
    def deposit_cash(self):
        """Wpłata gotówki do banku"""
        amount = input("💰 Deposit amount: $")
        try:
            amount = int(amount)
            if amount <= self.player["cash"]:
                self.player["cash"] -= amount
                self.player["bank_balance"] += amount
                print(f"✅ Deposited ${amount} to Cybernetic Vault")
            else:
                print("❌ Insufficient cash!")
        except ValueError:
            print("❌ Invalid amount!")
        input("Press Enter to continue...")
    
    def withdraw_cash(self):
        """Wypłata gotówki z banku"""
        amount = input("💸 Withdrawal amount: $")
        try:
            amount = int(amount)
            if amount <= self.player["bank_balance"]:
                self.player["bank_balance"] -= amount
                self.player["cash"] += amount
                print(f"✅ Withdrew ${amount} from Cybernetic Vault")
            else:
                print("❌ Insufficient bank balance!")
        except ValueError:
            print("❌ Invalid amount!")
        input("Press Enter to continue...")
    
    def furby_coin_exchange(self):
        """Wymiana FurbyCoin"""
        print(f"\n🪙 FURBYCOIN EXCHANGE")
        print(f"Your FurbyCoins: {self.player['furby_coins']}")
        print(f"Current rate: 1 FurbyCoin = $150")
        
        action = input("Action (buy/sell): ").lower()
        if action == "buy":
            usd_amount = int(input("USD to spend: $"))
            coins = usd_amount // 150
            if usd_amount <= self.player["cash"]:
                self.player["cash"] -= coins * 150
                self.player["furby_coins"] += coins
                print(f"✅ Bought {coins} FurbyCoins")
        elif action == "sell":
            coins = int(input("FurbyCoins to sell: "))
            if coins <= self.player["furby_coins"]:
                self.player["furby_coins"] -= coins
                self.player["cash"] += coins * 150
                print(f"✅ Sold {coins} FurbyCoins for ${coins * 150}")
        
        input("Press Enter to continue...")
    
    def encounter(self):
        """Spotkanie z NPC"""
        current_loc = self.world[self.position]
        npc_type = random.choice(current_loc["encounters"])
        
        if npc_type in self.npcs:
            npc = self.npcs[npc_type]
            
            self.clear_screen()
            self.draw_hud()
            
            print(f"\n💃 ENCOUNTER: {npc['name']} - {npc['style']}")
            print("\nActions:")
            print("1️⃣  😘 Flirt & Seduce")
            print("2️⃣  💰 Show Wealth")
            print("3️⃣  🔥 Physical Interaction")
            print("4️⃣  🎲 Cyber Gamble")
            print("5️⃣  🚗 Drive Away")
            
            choice = input("\nChoose action: ")
            
            if choice in ["1", "2", "3", "4"]:
                # Analiza interakcji przez MŚWR
                interaction = {
                    "type": f"social_interaction_{choice}",
                    "npc": npc['name'],
                    "player_state": self.player
                }
                
                analysis = self.cyber_bank.analyze_transaction(interaction)
                
                heat_gain = npc["heat_reward"]
                if analysis['probability_score'] > 0.8:
                    heat_gain = int(heat_gain * 1.5)  # Bonus za wysoką pewność
                    print(f"🧠 MŚWR Enhanced interaction! +50% heat bonus")
                
                self.player["heat"] += heat_gain
                self.player["reputation"] += random.randint(2, 8)
                
                responses = [
                    f"🔥 {npc['name']}: 'Your cyber-aura is intoxicating...'",
                    f"💋 {npc['name']}: 'In this digital realm, you're my favorite algorithm...'",
                    f"✨ {npc['name']}: 'Let's sync our neural networks...'"
                ]
                print(random.choice(responses))
                
            elif choice == "5":
                print("🚗 You drive away through the neon-lit streets...")
            
            time.sleep(2)
    
    def travel(self):
        """Podróżowanie po cyber-mieście"""
        print("\n🛣️ CYBER CITY NAVIGATION")
        print("⬅️ L: Left | ➡️ R: Right | ⬆️ F: Forward")
        
        choice = input("Direction: ").upper()
        
        if choice == "L" and self.position > 0:
            self.position -= 1
            print("⬅️ Moving left through the digital matrix...")
        elif choice == "R" and self.position < len(self.world) - 1:
            self.position += 1
            print("➡️ Cruising right into cyber-luxury...")
        elif choice == "F":
            self.random_event()
        else:
            print("🔄 No route available, staying in position...")
        
        self.player["energy"] -= random.randint(5, 15)
        time.sleep(1)
    
    def random_event(self):
        """Losowe wydarzenia"""
        events = [
            "🚨 Cyber Police scan - lay low or pay fine (-$200)!",
            "💎 Found crypto wallet: +$500!",
            "🔥 Viral post boost: +15 Heat!",
            "🌐 Network lag: lose turn.",
            "🎁 Mystery sponsor: +3 FurbyCoins!"
        ]
        
        event = random.choice(events)
        print(f"📡 {event}")
        
        if "fine" in event:
            self.player["cash"] = max(0, self.player["cash"] - 200)
        elif "+$500" in event:
            self.player["cash"] += 500
        elif "+15 Heat" in event:
            self.player["heat"] += 15
        elif "+3 FurbyCoins" in event:
            self.player["furby_coins"] += 3
    
    def check_game_over(self):
        """Sprawdzenie warunków końca gry"""
        if self.player["energy"] <= 0:
            print("😴 Energy depleted! Need to recharge in cyber-space...")
            return True
        if self.player["heat"] >= 100:
            print("🔥🔥 MAXIMUM HEAT ACHIEVED! You're the cyber-legend of the night!")
            print("🏆 VICTORY: Conquered the digital realm!")
            return True
        if self.player["cash"] <= 0 and self.player["bank_balance"] <= 0:
            print("💸 Bankrupt in cyber-space! Game over...")
            return True
        return False
    
    def start_game(self):
        """Główna pętla gry"""
        self.clear_screen()
        print("🎮 " + "="*60)
        print("  AI FURBY: CYBERNETIC BANK EDITION 1.25D")
        print("  San Andreas meets Conscious AI Trading")
        print("  Powered by MŚWR Technology")
        print("="*64)
        
        self.player["name"] = input("\n🤖 Enter your Cyber-Furby name: ") or "CyberAnon"
        
        print(f"\nWelcome to the digital underground, {self.player['name']}!")
        print("Cruise, seduce, trade, and dominate the cyber-city!")
        
        while not self.check_game_over():
            self.clear_screen()
            self.draw_hud()
            
            print("\n🎯 MAIN ACTIONS:")
            print("1️⃣  🚗 Cruise the Cyber-City")
            print("2️⃣  👀 Seek Cyber Encounters")
            print("3️⃣  💤 Rest & Recharge (Energy +50)")
            print("4️⃣  🏦 Access Cybernetic Bank")
            print("5️⃣  📊 View Stats & Analytics")
            
            choice = input("\n🤖 Select action: ")
            
            if choice == "1":
                self.travel()
            elif choice == "2":
                self.encounter()
            elif choice == "3":
                self.player["energy"] = min(100, self.player["energy"] + 50)
                print("😴 Recharging in virtual sanctuary...")
                time.sleep(1)
            elif choice == "4":
                self.cyber_bank_menu()
            elif choice == "5":
                self.show_detailed_stats()
            else:
                print("❓ Invalid action. Try again!")
                time.sleep(1)
        
        print(f"\n🏆 GAME OVER! Your legacy: Reputation {self.player['reputation']}")
        print("Thanks for playing AI Furby: Cybernetic Bank Edition!")
    
    def show_detailed_stats(self):
        """Pokazuje szczegółowe statystyki"""
        self.clear_screen()
        print("📊 DETAILED PLAYER ANALYTICS")
        print("="*50)
        print(f"💰 Financial Status:")
        print(f"   Cash: ${self.player['cash']}")
        print(f"   Bank: ${self.player['bank_balance']}")
        print(f"   Investments: ${self.player['investments']}")
        print(f"   FurbyCoins: {self.player['furby_coins']}")
        print(f"   Total Worth: ${self.player['cash'] + self.player['bank_balance'] + self.player['investments'] + self.player['furby_coins'] * 150}")
        
        print(f"\n🎮 Game Status:")
        print(f"   Heat Level: {self.player['heat']}/100")
        print(f"   Energy: {self.player['energy']}/100")
        print(f"   Reputation: {self.player['reputation']}")
        print(f"   Items Owned: {len(self.player['items'])}")
        
        print(f"\n🌍 World Status:")
        print(f"   Current Location: {self.world[self.position]['name']}")
        print(f"   Locations Explored: {min(self.position + 1, len(self.world))}/{len(self.world)}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        game = Furby125DCyberBank()
        game.start_game()
    except KeyboardInterrupt:
        print("\n\n🚪 Exiting cyber-space... Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("Please restart the game.")