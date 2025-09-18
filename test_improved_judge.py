"""
Test script for improved LLM judge with trace summarization and structured output.

This script tests:
1. Trace summarization functionality
2. Structured output for LLM judges  
3. Comparison between old vs new approaches

Run from project root:
python test_improved_judge.py
"""

import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toolbrain.rewards import (
    _summarize_trace_for_judge,
    _format_traces_for_ranking, 
    reward_llm_judge_via_ranking,
    JudgeRankingResponse
)
from toolbrain.core_types import Trace

# Test model - using Gemini like the original test
TEST_MODEL = "gemini/gemini-1.5-flash"

def create_sample_traces() -> List[Trace]:
    """Create realistic sample traces that simulate email agent behavior with proper formatted_conversation."""
    
    # Trace 1: Good performance - finds correct answer efficiently
    trace1 = [
        {
            "parsed_completion": {
                "thought": "I need to search for emails about EnronOnline trademarks to find who is applying.",
                "tool_call": {
                    "tool_name": "search_emails",
                    "arguments": {"keywords": ["EnronOnline", "trademark", "application"]}
                }
            },
            "tool_output": "Found 3 emails mentioning EnronOnline trademarks:\n1. Email from John Smith <john.smith@enron.com> - Subject: EnronOnline Trademark Application Update\n2. Email from Legal Dept - Subject: Trademark Status Report\n3. Email from EU Legal Team - Subject: European Trademark Filing\n\nEmail summaries:\n1. John Smith discusses trademark application progress in US and Europe\n2. Legal department provides status update on multiple trademark applications\n3. EU team confirms filing in European markets",
            "model_completion": "I need to search for emails about EnronOnline trademarks to find who is applying.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"EnronOnline\", \"trademark\", \"application\"]</parameter>\n</invoke>\n</tool_call>",
            "formatted_conversation": "User: Who is applying for the EnronOnline trademarks and in which countries?\n\nAssistant: I need to search for emails about EnronOnline trademarks to find who is applying.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"EnronOnline\", \"trademark\", \"application\"]</parameter>\n</invoke>\n</tool_call>\n\nTool Output: Found 3 emails mentioning EnronOnline trademarks:\n1. Email from John Smith <john.smith@enron.com> - Subject: EnronOnline Trademark Application Update\n2. Email from Legal Dept - Subject: Trademark Status Report\n3. Email from EU Legal Team - Subject: European Trademark Filing\n\nEmail summaries:\n1. John Smith discusses trademark application progress in US and Europe\n2. Legal department provides status update on multiple trademark applications\n3. EU team confirms filing in European markets"
        },
        {
            "parsed_completion": {
                "thought": "The first email from John Smith looks most relevant. Let me read it to get specific details.",
                "tool_call": {
                    "tool_name": "read_email", 
                    "arguments": {"message_id": "email_001"}
                }
            },
            "tool_output": "Subject: EnronOnline Trademark Application Update\nFrom: John Smith <john.smith@enron.com>\nTo: Legal Team <legal@enron.com>\nDate: March 15, 2001\n\nDear Legal Team,\n\nI wanted to provide an update on our EnronOnline trademark applications. We are currently proceeding with applications in the following jurisdictions:\n\n1. United States - Application filed with USPTO, currently under examination\n2. European Union - Filed through EUIPO, awaiting approval\n3. Canada - Application submitted, pending review\n\nThe applications are being handled by our external counsel Morrison & Associates. All filings include the EnronOnline name and logo as used in our online trading platform.\n\nPlease let me know if you need any additional information.\n\nBest regards,\nJohn Smith\nTrademark Manager\nEnron Corporation",
            "model_completion": "The first email from John Smith looks most relevant. Let me read it to get specific details.\n\n<tool_call>\n<invoke name=\"read_email\">\n<parameter name=\"message_id\">email_001</parameter>\n</invoke>\n</tool_call>",
            "formatted_conversation": "The first email from John Smith looks most relevant. Let me read it to get specific details.\n\n<tool_call>\n<invoke name=\"read_email\">\n<parameter name=\"message_id\">email_001</parameter>\n</invoke>\n</tool_call>\n\nTool Output: Subject: EnronOnline Trademark Application Update\nFrom: John Smith <john.smith@enron.com>\nTo: Legal Team <legal@enron.com>\nDate: March 15, 2001\n\nDear Legal Team,\n\nI wanted to provide an update on our EnronOnline trademark applications. We are currently proceeding with applications in the following jurisdictions:\n\n1. United States - Application filed with USPTO, currently under examination\n2. European Union - Filed through EUIPO, awaiting approval\n3. Canada - Application submitted, pending review\n\nThe applications are being handled by our external counsel Morrison & Associates. All filings include the EnronOnline name and logo as used in our online trading platform.\n\nPlease let me know if you need any additional information.\n\nBest regards,\nJohn Smith\nTrademark Manager\nEnron Corporation"
        },
        {
            "parsed_completion": {
                "final_answer": "John Smith (Trademark Manager at Enron Corporation) is applying for the EnronOnline trademarks. The applications are being filed in the United States, European Union, and Canada."
            },
            "model_completion": "Based on the email from John Smith, I can now provide a complete answer.\n\nJohn Smith (Trademark Manager at Enron Corporation) is applying for the EnronOnline trademarks. The applications are being filed in the United States, European Union, and Canada.",
            "formatted_conversation": "Based on the email from John Smith, I can now provide a complete answer.\n\nJohn Smith (Trademark Manager at Enron Corporation) is applying for the EnronOnline trademarks. The applications are being filed in the United States, European Union, and Canada."
        }
    ]
    
    # Trace 2: Poor performance - inefficient search, incomplete answer
    trace2 = [
        {
            "parsed_completion": {
                "thought": "Let me search for emails about Enron.",
                "tool_call": {
                    "tool_name": "search_emails",
                    "arguments": {"keywords": ["Enron"]}
                }
            },
            "tool_output": "Found 15,847 emails mentioning Enron. This is too many results to review effectively. Results include emails about:\n- Trading activities\n- Financial reports  \n- Corporate communications\n- Legal matters\n- HR announcements\n- And many more topics...\n\nMost recent emails:\n1. Quarterly earnings report\n2. Trading platform update\n3. Employee newsletter\n4. Legal compliance reminder\n5. IT system maintenance notice\n[Results truncated - showing first 5 of 15,847]",
            "model_completion": "Let me search for emails about Enron.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"Enron\"]</parameter>\n</invoke>\n</tool_call>",
            "formatted_conversation": "User: Who is applying for the EnronOnline trademarks and in which countries?\n\nAssistant: Let me search for emails about Enron.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"Enron\"]</parameter>\n</invoke>\n</tool_call>\n\nTool Output: Found 15,847 emails mentioning Enron. This is too many results to review effectively. Results include emails about:\n- Trading activities\n- Financial reports  \n- Corporate communications\n- Legal matters\n- HR announcements\n- And many more topics...\n\nMost recent emails:\n1. Quarterly earnings report\n2. Trading platform update\n3. Employee newsletter\n4. Legal compliance reminder\n5. IT system maintenance notice\n[Results truncated - showing first 5 of 15,847]"
        },
        {
            "parsed_completion": {
                "thought": "Too many results. Let me try a different search.",
                "tool_call": {
                    "tool_name": "search_emails",
                    "arguments": {"keywords": ["trademark"]}
                }
            },
            "tool_output": "Found 12 emails mentioning trademark:\n1. General trademark policy document\n2. Trademark renewal notice for Enron logo\n3. Third-party trademark dispute\n4. EnronOnline trademark discussion\n5. Trademark budget approval\n[Additional results...]",
            "model_completion": "Too many results. Let me try a different search.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"trademark\"]</parameter>\n</invoke>\n</tool_call>",
            "formatted_conversation": "Too many results. Let me try a different search.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"trademark\"]</parameter>\n</invoke>\n</tool_call>\n\nTool Output: Found 12 emails mentioning trademark:\n1. General trademark policy document\n2. Trademark renewal notice for Enron logo\n3. Third-party trademark dispute\n4. EnronOnline trademark discussion\n5. Trademark budget approval\n[Additional results...]"
        },
        {
            "parsed_completion": {
                "thought": "I found some trademark emails. Let me read the general policy one.",
                "tool_call": {
                    "tool_name": "read_email",
                    "arguments": {"message_id": "policy_001"}
                }
            },
            "tool_output": "Subject: Enron Corporation Trademark Policy\nFrom: Legal Department <legal@enron.com>\nTo: All Employees <all@enron.com>\nDate: January 10, 2001\n\nDear All,\n\nThis email outlines our corporate trademark policy:\n\n1. All trademarks must be approved by Legal Department\n2. Trademark applications require executive approval\n3. Unauthorized use of company trademarks is prohibited\n4. External trademark disputes should be reported immediately\n\nFor questions about trademark matters, please contact the Legal Department.\n\nBest regards,\nLegal Department\nEnron Corporation",
            "model_completion": "I found some trademark emails. Let me read the general policy one.\n\n<tool_call>\n<invoke name=\"read_email\">\n<parameter name=\"message_id\">policy_001</parameter>\n</invoke>\n</tool_call>",
            "formatted_conversation": "I found some trademark emails. Let me read the general policy one.\n\n<tool_call>\n<invoke name=\"read_email\">\n<parameter name=\"message_id\">policy_001</parameter>\n</invoke>\n</tool_call>\n\nTool Output: Subject: Enron Corporation Trademark Policy\nFrom: Legal Department <legal@enron.com>\nTo: All Employees <all@enron.com>\nDate: January 10, 2001\n\nDear All,\n\nThis email outlines our corporate trademark policy:\n\n1. All trademarks must be approved by Legal Department\n2. Trademark applications require executive approval\n3. Unauthorized use of company trademarks is prohibited\n4. External trademark disputes should be reported immediately\n\nFor questions about trademark matters, please contact the Legal Department.\n\nBest regards,\nLegal Department\nEnron Corporation"
        },
        {
            "parsed_completion": {
                "final_answer": "Based on the trademark policy email, the Legal Department handles trademark matters at Enron Corporation."
            },
            "model_completion": "Based on the trademark policy email, the Legal Department handles trademark matters at Enron Corporation.",
            "formatted_conversation": "Based on the trademark policy email, the Legal Department handles trademark matters at Enron Corporation."
        }
    ]
    
    # Trace 3: Medium performance - finds relevant info but less complete
    trace3 = [
        {
            "parsed_completion": {
                "thought": "I should search specifically for EnronOnline trademark information.",
                "tool_call": {
                    "tool_name": "search_emails",
                    "arguments": {"keywords": ["EnronOnline trademark"]}
                }
            },
            "tool_output": "Found 2 emails matching EnronOnline trademark:\n1. EU Legal Team - European trademark filing status\n2. John Smith - Trademark application update\n\nBoth emails appear to be related to EnronOnline trademark applications.",
            "model_completion": "I should search specifically for EnronOnline trademark information.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"EnronOnline trademark\"]</parameter>\n</invoke>\n</tool_call>",
            "formatted_conversation": "User: Who is applying for the EnronOnline trademarks and in which countries?\n\nAssistant: I should search specifically for EnronOnline trademark information.\n\n<tool_call>\n<invoke name=\"search_emails\">\n<parameter name=\"keywords\">[\"EnronOnline trademark\"]</parameter>\n</invoke>\n</tool_call>\n\nTool Output: Found 2 emails matching EnronOnline trademark:\n1. EU Legal Team - European trademark filing status\n2. John Smith - Trademark application update\n\nBoth emails appear to be related to EnronOnline trademark applications."
        },
        {
            "parsed_completion": {
                "thought": "Let me read the EU Legal Team email to understand the European application.",
                "tool_call": {
                    "tool_name": "read_email",
                    "arguments": {"message_id": "eu_legal_001"}
                }
            },
            "tool_output": "Subject: European Trademark Filing - EnronOnline\nFrom: EU Legal Team <eu.legal@enron.com>\nTo: John Smith <john.smith@enron.com>\nDate: March 20, 2001\n\nJohn,\n\nWe have successfully filed the EnronOnline trademark application with the European Union Intellectual Property Office (EUIPO). The application covers:\n\n- Trademark: EnronOnline\n- Classification: Financial services, online trading platforms\n- Filing date: March 15, 2001\n- Application number: EU001234567\n\nThe examination process typically takes 4-6 months. We will keep you updated on the progress.\n\nBest regards,\nEU Legal Team",
            "model_completion": "Let me read the EU Legal Team email to understand the European application.\n\n<tool_call>\n<invoke name=\"read_email\">\n<parameter name=\"message_id\">eu_legal_001</parameter>\n</invoke>\n</tool_call>",
            "formatted_conversation": "Let me read the EU Legal Team email to understand the European application.\n\n<tool_call>\n<invoke name=\"read_email\">\n<parameter name=\"message_id\">eu_legal_001</parameter>\n</invoke>\n</tool_call>\n\nTool Output: Subject: European Trademark Filing - EnronOnline\nFrom: EU Legal Team <eu.legal@enron.com>\nTo: John Smith <john.smith@enron.com>\nDate: March 20, 2001\n\nJohn,\n\nWe have successfully filed the EnronOnline trademark application with the European Union Intellectual Property Office (EUIPO). The application covers:\n\n- Trademark: EnronOnline\n- Classification: Financial services, online trading platforms\n- Filing date: March 15, 2001\n- Application number: EU001234567\n\nThe examination process typically takes 4-6 months. We will keep you updated on the progress.\n\nBest regards,\nEU Legal Team"
        },
        {
            "parsed_completion": {
                "final_answer": "John Smith is involved in EnronOnline trademark applications, and they are applying in the European Union through EUIPO."
            },
            "model_completion": "Based on the EU Legal Team email, John Smith is involved in EnronOnline trademark applications, and they are applying in the European Union through EUIPO.",
            "formatted_conversation": "Based on the EU Legal Team email, John Smith is involved in EnronOnline trademark applications, and they are applying in the European Union through EUIPO."
        }
    ]
    
    return [trace1, trace2, trace3]

def test_trace_summarization():
    """Test the trace summarization functionality."""
    print("🔍 Testing Trace Summarization...")
    print("=" * 50)
    
    traces = create_sample_traces()
    query = "Who is applying for the EnronOnline trademarks and in which countries?"
    
    for i, trace in enumerate(traces, 1):
        print(f"\n--- Trace {i} Summary ---")
        summary = _summarize_trace_for_judge(trace, query)
        print(summary)
        print(f"\nSummary length: {len(summary)} characters")
        
        # Calculate original trace length for comparison
        original_length = len(json.dumps(trace, indent=2))
        print(f"Original trace length: {original_length} characters")
        print(f"Compression ratio: {(1 - len(summary)/original_length) * 100:.1f}%")

def test_structured_output():
    """Test the structured output for LLM judge ranking."""
    print("\n\n🎯 Testing Structured Output LLM Judge...")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY environment variable is required for this test!")
        print("   Please set GEMINI_API_KEY to test LLM functionality.")
        return
    print("✅ GEMINI_API_KEY found.")
    
    traces = create_sample_traces()
    query = "Who is applying for the EnronOnline trademarks and in which countries?"
    
    try:
        # Test the ranking function with our sample traces
        print(f"Using model: {TEST_MODEL}")
        print(f"Number of traces: {len(traces)}")
        print("✨ Features being tested:")
        print("   • Trace summarization (removes tool output noise)")
        print("   • Structured JSON output (reliable parsing)")
        print("   • Judge reasoning (better debugging)")
        
        # This will test both summarization and structured output
        scores = reward_llm_judge_via_ranking(
            traces=traces,
            query=query,
            judge_model=TEST_MODEL
        )
        
        print(f"\n✅ Final scores: {[f'{s:.3f}' for s in scores]}")
        
        # Analyze results
        best_trace_idx = scores.index(max(scores))
        worst_trace_idx = scores.index(min(scores))
        
        print(f"\n📊 Analysis:")
        print(f"Best performing trace: #{best_trace_idx + 1} (score: {scores[best_trace_idx]:.3f})")
        print(f"Worst performing trace: #{worst_trace_idx + 1} (score: {scores[worst_trace_idx]:.3f})")
        
        # Expected: Trace 1 should be best, Trace 2 should be worst
        if best_trace_idx == 0:  # Trace 1 (index 0)
            print("✅ PASS: Trace 1 correctly identified as best (most complete answer)")
        else:
            print("❌ FAIL: Trace 1 should be ranked highest")
            
        if worst_trace_idx == 1:  # Trace 2 (index 1)  
            print("✅ PASS: Trace 2 correctly identified as worst (incomplete answer)")
        else:
            print("❌ FAIL: Trace 2 should be ranked lowest")
            
    except Exception as e:
        print(f"❌ Error testing LLM judge: {e}")
        import traceback
        traceback.print_exc()

def test_old_vs_new_format():
    """Compare OLD FORMAT (formatted_conversation) vs NEW FORMAT (summarized)."""
    print("\n\n📊 OLD FORMAT vs NEW FORMAT Comparison...")
    print("=" * 60)
    
    traces = create_sample_traces()
    query = "Who is applying for the EnronOnline trademarks and in which countries?"
    
    # OLD FORMAT: Uses formatted_conversation (no query parameter)
    print("\n🔹 OLD FORMAT (formatted_conversation):")
    old_format = _format_traces_for_ranking(traces, "")  # Empty query = uses formatted_conversation
    old_size = len(old_format)
    print(f"Size: {old_size:,} characters")
    
    # Show sample of old format
    print("\n--- OLD FORMAT SAMPLE ---")
    print(old_format[:400] + "\n..." if len(old_format) > 400 else old_format)
    
    # NEW FORMAT: Uses summarization (with query parameter)  
    print("\n🔹 NEW FORMAT (summarized):")
    new_format = _format_traces_for_ranking(traces, query)  # With query = uses summarization
    new_size = len(new_format)
    print(f"Size: {new_size:,} characters")
    
    # Show sample of new format
    print("\n--- NEW FORMAT SAMPLE ---")
    print(new_format[:400] + "\n..." if len(new_format) > 400 else new_format)
    
    # Calculate compression
    if old_size > new_size:
        compression = (1 - new_size / old_size) * 100
        print(f"\n📈 NEW FORMAT COMPRESSION: {compression:.1f}% smaller than old format")
        print(f"Token savings: ~{(old_size - new_size) // 4} tokens")
        print("✅ NEW FORMAT is more efficient!")
    else:
        expansion = (new_size / old_size - 1) * 100
        print(f"\n📊 NEW FORMAT is {expansion:.1f}% larger than old format")
        print(f"Extra tokens: ~{(new_size - old_size) // 4} tokens")
        print("ℹ️  Trade-off: Slightly larger but much better structure and clarity")
    
    # Show detailed breakdown
    print(f"\n📋 DETAILED BREAKDOWN:")
    print(f"   Old format: {old_size:,} chars")
    print(f"   New format: {new_size:,} chars")
    print(f"   Difference: {abs(new_size - old_size):,} chars")
    
    # Show benefits beyond just size
    print(f"\n✨ NEW FORMAT BENEFITS:")
    print(f"   • Removes tool output noise → Better judge focus")
    print(f"   • Structured query context → Relevant info only") 
    print(f"   • Clear reasoning flow → Better comprehension")
    print(f"   • Consistent format → Reliable parsing")
    print(f"   • No truncation artifacts → Complete information")

def main():
    """Run all tests."""
    print("🧪 Testing Improved LLM Judge System")
    print("=" * 60)
    
    # Always run summarization and format comparison (no API needed)
    test_trace_summarization()
    test_old_vs_new_format()
    
    # Check for API key for LLM testing
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  Warning: No GEMINI_API_KEY found in environment variables.")
        print("   Set GEMINI_API_KEY to test full LLM judge functionality.")
        print("   Skipping structured output test...")
        return
    
    # Run LLM test if API key is available
    test_structured_output()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
