"""
Comprehensive system test script.
Tests all major components to verify the system is working.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Dict, List
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemTester:
    """Test all major system components."""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
    
    def test_imports(self) -> bool:
        """Test that all core modules can be imported."""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Module Imports")
        logger.info("="*60)
        
        modules = {
            'Data Loader': 'src.data.data_loader',
            'Dataset Config': 'src.data.dataset_config',
            'Embeddings': 'src.models.embeddings',
            'Classifier': 'src.models.classifier',
            'DS Mass Function': 'src.models.ds_mass_function',
            'Customer Agent': 'src.agents.customer_agent',
            'Metrics': 'src.utils.metrics',
            'File IO': 'src.utils.file_io',
            'Explainability': 'src.utils.explainability',
            'Faithfulness': 'src.utils.faithfulness',
            'Evaluation Curves': 'src.utils.evaluation_curves',
            'Query Selector': 'src.utils.query_selector',
            'User Study': 'src.utils.user_study',
            'Hierarchy Loader': 'config.hierarchy_loader',
            'Threshold Loader': 'config.threshold_loader'
        }
        
        all_passed = True
        for name, module_path in modules.items():
            try:
                __import__(module_path)
                logger.info(f"✅ {name}: OK")
                self.results[f"Import: {name}"] = "✅ PASS"
            except Exception as e:
                logger.error(f"❌ {name}: FAILED - {str(e)}")
                self.results[f"Import: {name}"] = f"❌ FAIL: {str(e)}"
                all_passed = False
        
        if all_passed:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_passed
    
    def test_config_files(self) -> bool:
        """Test configuration files exist."""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Configuration Files")
        logger.info("="*60)
        
        config_files = {
            'Banking77 Hierarchy': 'config/hierarchies/banking77_hierarchy.json',
            'Banking77 Intents': 'config/hierarchies/banking77_intents.json',
            'Hierarchy Example': 'config/banking77_hierarchy.example.json',
            'Intents Example': 'config/banking77_intents.example.json',
            'Thresholds Example': 'config/thresholds.example.json'
        }
        
        all_passed = True
        for name, filepath in config_files.items():
            full_path = Path(__file__).parent.parent / filepath
            if full_path.exists():
                logger.info(f"✅ {name}: Found")
                self.results[f"Config: {name}"] = "✅ PASS"
            else:
                logger.error(f"❌ {name}: Missing at {filepath}")
                self.results[f"Config: {name}"] = f"❌ FAIL: Missing"
                all_passed = False
        
        if all_passed:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_passed
    
    def test_hierarchy_loading(self) -> bool:
        """Test hierarchy loading functionality."""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Hierarchy Loading")
        logger.info("="*60)
        
        try:
            from config.hierarchy_loader import (
                load_hierarchy_from_json,
                load_hierarchical_intents_from_json
            )
            
            # Test hierarchy loading
            hierarchy_path = Path(__file__).parent.parent / 'config/hierarchies/banking77_hierarchy.json'
            hierarchy = load_hierarchy_from_json(str(hierarchy_path))
            
            logger.info(f"✅ Hierarchy loaded: {len(hierarchy)} nodes")
            
            # Test intents loading
            intents_path = Path(__file__).parent.parent / 'config/hierarchies/banking77_intents.json'
            intents = load_hierarchical_intents_from_json(str(intents_path))
            
            logger.info(f"✅ Intents loaded: {len(intents)} intents")
            
            # Validate structure
            if 'Card Issues' in hierarchy and 'activate_my_card' in intents:
                logger.info("✅ Hierarchy structure valid")
                self.results["Hierarchy Loading"] = "✅ PASS"
                self.passed += 1
                return True
            else:
                logger.error("❌ Invalid hierarchy structure")
                self.results["Hierarchy Loading"] = "❌ FAIL: Invalid structure"
                self.failed += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Hierarchy loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["Hierarchy Loading"] = f"❌ FAIL: {str(e)}"
            self.failed += 1
            return False
    
    def test_embeddings(self) -> bool:
        """Test embedding generation."""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Embeddings")
        logger.info("="*60)
        
        try:
            from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
            from config.hierarchy_loader import load_hierarchical_intents_from_json
            
            # Test sentence embedder
            embedder = SentenceEmbedder()
            test_text = "I want to activate my card"
            embedding = embedder.get_embedding(test_text)
            
            logger.info(f"✅ Sentence embedding generated: shape {embedding.shape}")
            
            # Test intent embeddings
            intents_path = Path(__file__).parent.parent / 'config/hierarchies/banking77_intents.json'
            intents = load_hierarchical_intents_from_json(str(intents_path))
            
            intent_embeddings = IntentEmbeddings(intents, embedder=embedder)
            all_embeddings = intent_embeddings.get_all_embeddings()
            
            logger.info(f"✅ Intent embeddings generated: {len(all_embeddings)} intents")
            
            self.results["Embeddings"] = "✅ PASS"
            self.passed += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Embeddings failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["Embeddings"] = f"❌ FAIL: {str(e)}"
            self.failed += 1
            return False
    
    def test_classifier_creation(self) -> bool:
        """Test classifier can be instantiated."""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Classifier Creation")
        logger.info("="*60)
        
        try:
            from src.models.classifier import IntentClassifier
            
            # Create classifier (without training)
            classifier = IntentClassifier(model_type='logistic')
            
            logger.info(f"✅ Classifier created: {type(classifier.model).__name__}")
            
            self.results["Classifier Creation"] = "✅ PASS"
            self.passed += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Classifier creation failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["Classifier Creation"] = f"❌ FAIL: {str(e)}"
            self.failed += 1
            return False
    
    def test_ds_mass_function(self) -> bool:
        """Test DS Mass Function can be instantiated."""
        logger.info("\n" + "="*60)
        logger.info("TEST 6: DS Mass Function")
        logger.info("="*60)
        
        try:
            from src.models.ds_mass_function import DSMassFunction
            from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
            from src.models.classifier import IntentClassifier
            from config.hierarchy_loader import (
                load_hierarchy_from_json,
                load_hierarchical_intents_from_json
            )
            
            # Load configuration
            hierarchy_path = Path(__file__).parent.parent / 'config/hierarchies/banking77_hierarchy.json'
            intents_path = Path(__file__).parent.parent / 'config/hierarchies/banking77_intents.json'
            
            hierarchy = load_hierarchy_from_json(str(hierarchy_path))
            intents = load_hierarchical_intents_from_json(str(intents_path))
            
            # Create components
            embedder = SentenceEmbedder()
            intent_embeddings = IntentEmbeddings(intents, embedder=embedder)
            classifier = IntentClassifier(model_type='logistic')
            
            # Create DS Mass Function
            ds_calculator = DSMassFunction(
                intent_embeddings=intent_embeddings.get_all_embeddings(),
                hierarchy=hierarchy,
                classifier=classifier
            )
            
            logger.info("✅ DS Mass Function created successfully")
            
            # Test basic methods
            test_leaf = 'activate_my_card'
            is_leaf = ds_calculator.is_leaf(test_leaf)
            logger.info(f"✅ is_leaf('{test_leaf}'): {is_leaf}")
            
            threshold = ds_calculator.get_confidence_threshold(test_leaf)
            logger.info(f"✅ get_confidence_threshold('{test_leaf}'): {threshold}")
            
            self.results["DS Mass Function"] = "✅ PASS"
            self.passed += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ DS Mass Function failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["DS Mass Function"] = f"❌ FAIL: {str(e)}"
            self.failed += 1
            return False
    
    def test_explainability(self) -> bool:
        """Test explainability modules."""
        logger.info("\n" + "="*60)
        logger.info("TEST 7: Explainability Modules")
        logger.info("="*60)
        
        try:
            from src.utils.explainability import BeliefTracker, BeliefVisualizer
            
            # Test BeliefTracker
            tracker = BeliefTracker()
            test_belief = {'intent1': 0.6, 'intent2': 0.3, 'intent3': 0.1}
            tracker.record_belief(test_belief, "Test Turn")
            
            logger.info("✅ BeliefTracker created and tested")
            
            # Test BeliefVisualizer
            visualizer = BeliefVisualizer()
            logger.info("✅ BeliefVisualizer created")
            
            self.results["Explainability"] = "✅ PASS"
            self.passed += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Explainability failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["Explainability"] = f"❌ FAIL: {str(e)}"
            self.failed += 1
            return False
    
    def test_faithfulness_validator(self) -> bool:
        """Test faithfulness validation module."""
        logger.info("\n" + "="*60)
        logger.info("TEST 8: Faithfulness Validator")
        logger.info("="*60)
        
        try:
            from src.utils.faithfulness import FaithfulnessValidator
            
            validator = FaithfulnessValidator()
            logger.info("✅ FaithfulnessValidator created")
            
            self.results["Faithfulness Validator"] = "✅ PASS"
            self.passed += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Faithfulness Validator failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["Faithfulness Validator"] = f"❌ FAIL: {str(e)}"
            self.failed += 1
            return False
    
    def test_acc_analyzer(self) -> bool:
        """Test ACC curve analyzer."""
        logger.info("\n" + "="*60)
        logger.info("TEST 9: ACC Curve Analyzer")
        logger.info("="*60)
        
        try:
            from src.utils.evaluation_curves import AccuracyCoverageBurdenAnalyzer
            
            analyzer = AccuracyCoverageBurdenAnalyzer()
            logger.info("✅ AccuracyCoverageBurdenAnalyzer created")
            
            self.results["ACC Analyzer"] = "✅ PASS"
            self.passed += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ ACC Analyzer failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["ACC Analyzer"] = f"❌ FAIL: {str(e)}"
            self.failed += 1
            return False
    
    def test_scripts_exist(self) -> bool:
        """Test that all main scripts exist."""
        logger.info("\n" + "="*60)
        logger.info("TEST 10: Scripts Existence")
        logger.info("="*60)
        
        scripts = {
            'Training': 'scripts/training/train.py',
            'Evaluation': 'scripts/evaluation/evaluate.py',
            'Compute Thresholds': 'scripts/evaluation/compute_thresholds.py',
            'User Study': 'scripts/user_study/run_user_study.py',
            'Simulated User': 'scripts/user_study/run_simulated_user.py',
            'Compare Users': 'scripts/user_study/compare_simulated_vs_real.py',
            'Select Queries': 'scripts/user_study/select_user_study_queries.py',
            'Test Faithfulness': 'scripts/analysis/test_faithfulness.py',
            'Analyze ACC': 'scripts/analysis/analyze_acc_curves.py'
        }
        
        all_passed = True
        for name, filepath in scripts.items():
            full_path = Path(__file__).parent.parent / filepath
            if full_path.exists():
                logger.info(f"✅ {name}: Found")
                self.results[f"Script: {name}"] = "✅ PASS"
            else:
                logger.error(f"❌ {name}: Missing at {filepath}")
                self.results[f"Script: {name}"] = f"❌ FAIL: Missing"
                all_passed = False
        
        if all_passed:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_passed
    
    def generate_report(self):
        """Generate final test report."""
        logger.info("\n" + "="*60)
        logger.info("SYSTEM TEST REPORT")
        logger.info("="*60)
        
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        logger.info(f"\nTotal Tests: {total}")
        logger.info(f"Passed: {self.passed}")
        logger.info(f"Failed: {self.failed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed > 0:
            logger.info("\n❌ FAILED TESTS:")
            for test_name, result in self.results.items():
                if "❌" in result:
                    logger.info(f"  - {test_name}: {result}")
        
        logger.info("\n" + "="*60)
        
        if self.failed == 0:
            logger.info("✅ ALL TESTS PASSED - System is ready!")
        else:
            logger.info(f"⚠️  {self.failed} test(s) failed - Review errors above")
        
        logger.info("="*60)
        
        return self.failed == 0


def main():
    """Run all system tests."""
    print("\n" + "="*60)
    print("DS PROJECT - COMPREHENSIVE SYSTEM TEST")
    print("="*60)
    
    tester = SystemTester()
    
    # Run all tests
    tester.test_imports()
    tester.test_config_files()
    tester.test_hierarchy_loading()
    tester.test_embeddings()
    tester.test_classifier_creation()
    tester.test_ds_mass_function()
    tester.test_explainability()
    tester.test_faithfulness_validator()
    tester.test_acc_analyzer()
    tester.test_scripts_exist()
    
    # Generate report
    all_passed = tester.generate_report()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
