"""
Grammar Checker Module for Universal Dependencies
Validates parsed sentences for morphological agreement and syntactic consistency

Works across all languages with CoNLL-U format and morphology tables
"""

import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GrammarError:
    """Represents a grammar error found in parsed text"""
    token_id: int
    token: str
    error_type: str
    description: str
    expected_feats: Optional[str] = None
    found_feats: Optional[str] = None
    related_token_id: Optional[int] = None
    related_token: Optional[str] = None
    suggestion: Optional[str] = None


class GrammarChecker:
    """
    Rule-based grammar checker using dependency parse output
    
    Validates:
    - Subject-verb agreement (number, person, gender)
    - Noun-adjective agreement
    - Determiner-noun agreement
    - Case agreement based on dependency relations
    """
    
    def __init__(self, morphology_table: Optional[Dict] = None):
        """
        Initialize grammar checker
        
        Args:
            morphology_table: Optional morphology lookup table for suggestions
                              Structure: {(lemma, upos, feats): [surface_forms]}
        """
        self.morphology_table = morphology_table
        self.has_suggestions = morphology_table is not None
    
    @staticmethod
    def load_morphology_table(table_path: str) -> Optional[Dict]:
        """Load morphology table from pickle file"""
        try:
            with open(table_path, 'rb') as f:
                result = pickle.load(f)
                # Extract table if wrapped in dict with stats
                if isinstance(result, dict) and 'table' in result:
                    return result['table']
                return result
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"⚠️  Error loading morphology table: {e}")
            return None
    
    def parse_feats(self, feats_str: str) -> Dict[str, str]:
        """
        Parse FEATS string into dictionary
        
        Args:
            feats_str: Feature string like "Number=Sing|Person=3"
        
        Returns:
            Dictionary like {'Number': 'Sing', 'Person': '3'}
        """
        if not feats_str or feats_str == '_':
            return {}
        
        feats_dict = {}
        for feat in feats_str.split('|'):
            if '=' in feat:
                key, value = feat.split('=', 1)
                feats_dict[key] = value
        
        return feats_dict
    
    def check_agreement(self, head_feats: Dict, dep_feats: Dict, 
                       agreement_features: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if features agree between head and dependent
        
        Args:
            head_feats: Feature dict of head word
            dep_feats: Feature dict of dependent word
            agreement_features: List of features to check (e.g., ['Number', 'Gender'])
        
        Returns:
            (agrees: bool, mismatched_features: List[str])
        """
        mismatched = []
        
        for feat in agreement_features:
            head_value = head_feats.get(feat)
            dep_value = dep_feats.get(feat)
            
            # Only check if both have the feature
            if head_value and dep_value and head_value != dep_value:
                mismatched.append(feat)
        
        return len(mismatched) == 0, mismatched
    
    def check_subject_verb_agreement(self, results: Dict) -> List[GrammarError]:
        """Check subject-verb agreement"""
        errors = []
        
        for i in range(len(results['tokens'])):
            deprel = results['deprel'][i]
            
            # Find subject relations
            if deprel in ['nsubj', 'nsubj:pass', 'csubj', 'csubj:pass']:
                subject_id = i
                verb_id = results['heads'][i] - 1  # Convert to 0-indexed
                
                if verb_id < 0 or verb_id >= len(results['tokens']):
                    continue
                
                # Check if head is a verb
                verb_upos = results['upos'][verb_id]
                if verb_upos not in ['VERB', 'AUX']:
                    continue
                
                # Parse features
                subject_feats = self.parse_feats(results['feats'][subject_id])
                verb_feats = self.parse_feats(results['feats'][verb_id])
                
                # Check agreement on Number, Person (Gender in some languages)
                agreement_features = ['Number', 'Person', 'Gender']
                agrees, mismatched = self.check_agreement(
                    subject_feats, verb_feats, agreement_features
                )
                
                if not agrees:
                    # Build error description
                    mismatch_details = []
                    for feat in mismatched:
                        subj_val = subject_feats.get(feat, '?')
                        verb_val = verb_feats.get(feat, '?')
                        mismatch_details.append(f"{feat}: subject={subj_val}, verb={verb_val}")
                    
                    error = GrammarError(
                        token_id=verb_id + 1,
                        token=results['tokens'][verb_id],
                        error_type="Subject-Verb Agreement",
                        description=f"Verb does not agree with subject in {', '.join(mismatched)}",
                        expected_feats=results['feats'][subject_id],
                        found_feats=results['feats'][verb_id],
                        related_token_id=subject_id + 1,
                        related_token=results['tokens'][subject_id]
                    )
                    
                    # Try to find suggestion
                    if self.has_suggestions:
                        # We would need lemma to look up, which parser doesn't predict
                        # This is a limitation - suggestions work best with lemmatizer
                        pass
                    
                    errors.append(error)
        
        return errors
    
    def check_noun_modifier_agreement(self, results: Dict) -> List[GrammarError]:
        """Check noun-adjective and noun-determiner agreement"""
        errors = []
        
        for i in range(len(results['tokens'])):
            deprel = results['deprel'][i]
            
            # Check adjective-noun agreement
            if deprel == 'amod':
                adj_id = i
                noun_id = results['heads'][i] - 1
                
                if noun_id < 0 or noun_id >= len(results['tokens']):
                    continue
                
                adj_feats = self.parse_feats(results['feats'][adj_id])
                noun_feats = self.parse_feats(results['feats'][noun_id])
                
                # Check Number, Gender, Case
                agreement_features = ['Number', 'Gender', 'Case']
                agrees, mismatched = self.check_agreement(
                    noun_feats, adj_feats, agreement_features
                )
                
                if not agrees:
                    error = GrammarError(
                        token_id=adj_id + 1,
                        token=results['tokens'][adj_id],
                        error_type="Adjective-Noun Agreement",
                        description=f"Adjective does not agree with noun in {', '.join(mismatched)}",
                        expected_feats=results['feats'][noun_id],
                        found_feats=results['feats'][adj_id],
                        related_token_id=noun_id + 1,
                        related_token=results['tokens'][noun_id]
                    )
                    errors.append(error)
            
            # Check determiner-noun agreement
            elif deprel == 'det':
                det_id = i
                noun_id = results['heads'][i] - 1
                
                if noun_id < 0 or noun_id >= len(results['tokens']):
                    continue
                
                det_feats = self.parse_feats(results['feats'][det_id])
                noun_feats = self.parse_feats(results['feats'][noun_id])
                
                # Check Number, Gender, Case
                agreement_features = ['Number', 'Gender', 'Case']
                agrees, mismatched = self.check_agreement(
                    noun_feats, det_feats, agreement_features
                )
                
                if not agrees:
                    error = GrammarError(
                        token_id=det_id + 1,
                        token=results['tokens'][det_id],
                        error_type="Determiner-Noun Agreement",
                        description=f"Determiner does not agree with noun in {', '.join(mismatched)}",
                        expected_feats=results['feats'][noun_id],
                        found_feats=results['feats'][det_id],
                        related_token_id=noun_id + 1,
                        related_token=results['tokens'][noun_id]
                    )
                    errors.append(error)
        
        return errors
    
    def check_case_agreement(self, results: Dict) -> List[GrammarError]:
        """Check if arguments have appropriate case for their syntactic function"""
        errors = []
        
        # Expected cases for common dependency relations
        expected_cases = {
            'nsubj': ['Nom'],  # Nominative for subjects
            'obj': ['Acc'],    # Accusative for objects
            'iobj': ['Dat'],   # Dative for indirect objects
            'obl': ['Loc', 'Ins', 'Abl', 'Gen'],  # Various for obliques
        }
        
        for i in range(len(results['tokens'])):
            deprel = results['deprel'][i]
            
            if deprel in expected_cases:
                feats = self.parse_feats(results['feats'][i])
                actual_case = feats.get('Case')
                
                if actual_case and actual_case not in expected_cases[deprel]:
                    error = GrammarError(
                        token_id=i + 1,
                        token=results['tokens'][i],
                        error_type="Case Agreement",
                        description=f"{deprel} typically requires {'/'.join(expected_cases[deprel])} case, found {actual_case}",
                        expected_feats=f"Case={expected_cases[deprel][0]}",
                        found_feats=f"Case={actual_case}"
                    )
                    errors.append(error)
        
        return errors
    
    def validate(self, results: Dict) -> List[GrammarError]:
        """
        Run all grammar checks on parsed sentence
        
        Args:
            results: Dictionary from predict_sentence() with keys:
                     tokens, upos, xpos, deprel, feats, heads
        
        Returns:
            List of GrammarError objects
        """
        all_errors = []
        
        # Run all checks
        all_errors.extend(self.check_subject_verb_agreement(results))
        all_errors.extend(self.check_noun_modifier_agreement(results))
        all_errors.extend(self.check_case_agreement(results))
        
        return all_errors


def format_grammar_errors(errors: List[GrammarError], results: Dict) -> str:
    """
    Format grammar errors for display
    
    Args:
        errors: List of GrammarError objects
        results: Parse results for context
    
    Returns:
        Formatted string for display
    """
    if not errors:
        return "\n✓ No grammar errors detected\n"
    
    output = []
    output.append("\n" + "="*70)
    output.append(f"GRAMMAR CHECK: Found {len(errors)} error(s)")
    output.append("="*70)
    
    for i, error in enumerate(errors, 1):
        output.append(f"\n❌ Error {i}: {error.error_type}")
        output.append(f"   Token {error.token_id}: \"{error.token}\"")
        output.append(f"   Issue: {error.description}")
        
        if error.related_token:
            output.append(f"   Related to token {error.related_token_id}: \"{error.related_token}\"")
        
        if error.expected_feats and error.found_feats:
            output.append(f"   Expected: {error.expected_feats}")
            output.append(f"   Found: {error.found_feats}")
        
        if error.suggestion:
            output.append(f"   💡 Suggestion: \"{error.suggestion}\"")
    
    output.append("\n" + "="*70)
    
    return "\n".join(output)


def print_grammar_report(errors: List[GrammarError], results: Dict):
    """Print formatted grammar error report"""
    print(format_grammar_errors(errors, results))
