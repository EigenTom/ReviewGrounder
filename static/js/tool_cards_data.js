// ReviewGrounder Featured Agents Data

const toolsData = [
  {
    id: 'drafter',
    name: 'Review Drafter',
    icon: '✍️',
    description: 'A fine-tuned LLM that generates an initial draft review from the paper text. It captures basic structure and stylistic conventions commonly observed in human-written reviews (section-level organization, syntax-level cues). Trained on DeepReview-13K, the Drafter primarily acquires shallow review patterns rather than grounded reasoning.',
    inputs: [
      { name: 'paper', type: 'str', desc: 'The full paper text (PDF converted to plain text)' }
    ],
    output: { type: 'str', desc: 'Initial draft review r⁽⁰⁾ with Summary, Strengths, Weaknesses, Questions' },
    demoCommands: [
      {
        command: 'draft = Drafter(p).generate()',
        description: 'Generate an initial draft review from the submission paper.'
      }
    ],
    limitations: [
      'Produces shallow drafts that often lack comparative context and evidential support.',
      'Acquires patterns from training data rather than deep paper understanding.'
    ],
    bestPractices: [
      'Use as the first stage—drafts are refined by grounding agents.',
      'Training data is disjoint from evaluation benchmark.'
    ],
    llmRequired: true,
    codeLink: '#'
  },
  {
    id: 'literature_searcher',
    name: 'Literature Searcher',
    icon: '🔍',
    description: 'A tool-integrated agent that situates the submission within contemporary literature. It extracts representative keywords from the title, abstract, and related work; queries the Semantic Scholar API for candidate papers (2023+); reranks results with OpenScholar-Reranker; and produces structured debriefs for the top-10 papers summarizing methodology, findings, and relevance to the target paper. This supports grounded novelty assessment and comparative analysis.',
    inputs: [
      { name: 'paper_sections', type: 'str', desc: 'Title, abstract, and related work of the submission' }
    ],
    output: { type: 'dict', desc: 'Structured debriefs of top-K relevant papers with methodology and findings' },
    demoCommands: [
      {
        command: 'keywords = extract_keywords(paper.title, paper.abstract, paper.related_work)\nresults = semantic_scholar.search(keywords)\ndebriefs = summarize_top_papers(rerank(results, top_k=10))',
        description: 'Retrieve and summarize related work for comparative analysis.'
      }
    ],
    limitations: [
      'Limited to Semantic Scholar corpus; may miss very recent or niche work.',
      'Reranker quality affects relevance of selected papers.'
    ],
    bestPractices: [
      '10 reranked papers per keyword yields optimal balance (ablation in paper).',
      'Use domain-specific rerankers like OpenScholar-Reranker for scientific literature.'
    ],
    llmRequired: true,
    codeLink: '#'
  },
  {
    id: 'insight_miner',
    name: 'Insight Miner',
    icon: '💡',
    description: 'Targets the conceptual and methodological core of the paper. It retrieves sections relevant to the technical approach, distills central contributions, and evaluates the validity of novelty claims and stated differences. Based on this analysis, it refines the method-focused parts of the draft by providing actionable suggestions grounded in specific parts of the paper (sections, formulas). Enhances accuracy of discussions around model design, algorithmic formulation, and implementation.',
    inputs: [
      { name: 'paper', type: 'str', desc: 'Full paper text' },
      { name: 'draft_weaknesses', type: 'str', desc: 'Method-related sections of the initial draft' }
    ],
    output: { type: 'str', desc: 'Refined method-focused critiques with evidence and actionable suggestions' },
    demoCommands: [
      {
        command: 'method_analysis = InsightMiner(paper).analyze_contributions()\nrefined = refine_draft(draft_weaknesses, method_analysis)',
        description: 'Verify methodology and consolidate evidence-based critiques.'
      }
    ],
    limitations: [
      'Relies on LLM ability to accurately parse technical sections.',
      'May miss subtle methodological issues.'
    ],
    bestPractices: [
      'Anchor critiques to specific sections, equations, or figures.',
      'Transform vague claims into precise, evidence-supported critiques.'
    ],
    llmRequired: true,
    codeLink: '#'
  },
  {
    id: 'result_analyzer',
    name: 'Result Analyzer',
    icon: '📊',
    description: 'Focuses exclusively on empirical evaluation. It extracts key experimental elements: datasets, baselines, evaluation metrics, performance gains, and statistical comparisons. Using these signals, it refines the experiment-related components of the draft, ensuring that claims about performance and effectiveness are faithful to reported results and grounded in concrete tables, figures, and quantitative comparisons. Complements the Insight Miner by strengthening empirical grounding.',
    inputs: [
      { name: 'paper', type: 'str', desc: 'Full paper text' },
      { name: 'draft_experiments', type: 'str', desc: 'Experiment-related sections of the initial draft' }
    ],
    output: { type: 'str', desc: 'Refined experiment critiques with quantitative evidence' },
    demoCommands: [
      {
        command: 'exp_elements = ResultAnalyzer(paper).extract(datasets, baselines, metrics)\nrefined = refine_experiment_section(draft_experiments, exp_elements)',
        description: 'Extract and validate experimental claims against paper evidence.'
      }
    ],
    limitations: [
      'Depends on structured extraction of tables and figures.',
      'Statistical significance may require external verification.'
    ],
    bestPractices: [
      'Reference specific tables and figures when discussing results.',
      'Ensure claims align with reported numbers and comparisons.'
    ],
    llmRequired: true,
    codeLink: '#'
  },
  {
    id: 'aggregator',
    name: 'Review Aggregator',
    icon: '📋',
    description: 'Synthesizes the outputs of all upstream agents to produce the final coherent, accurate, and actionable review. Takes as input the paper, initial draft, grounded representation E(p), and meta-rubrics R^meta. Performs final consolidation: (1) corrects factual errors; (2) strengthens critiques with paper-grounded evidence; (3) translates observations into clear, constructive suggestions; (4) produces balanced assessment aligning with rubric expectations on coverage, clarity, and tone. Paper-specific rubrics are NOT exposed at generation time.',
    inputs: [
      { name: 'paper', type: 'str', desc: 'Full paper text' },
      { name: 'draft', type: 'str', desc: 'Initial draft review r⁽⁰⁾' },
      { name: 'grounded_evidence', type: 'dict', desc: 'E(p) from Literature Searcher, Insight Miner, Result Analyzer' },
      { name: 'meta_rubrics', type: 'list', desc: 'R^meta: 8 evaluation dimensions' }
    ],
    output: { type: 'str', desc: 'Final review with Summary, Strengths, Weaknesses, Questions, Rating, Decision' },
    demoCommands: [
      {
        command: 'final_review = Aggregator(paper, draft, E, R_meta).synthesize()',
        description: 'Synthesize draft and evidence into coherent final review.'
      }
    ],
    limitations: [
      'Quality depends on upstream agent outputs.',
      'Must balance comprehensiveness with conciseness.'
    ],
    bestPractices: [
      'Ensure all critiques are evidence-grounded.',
      'Maintain constructive, professional tone.',
      'Cover all major aspects: methodology, experiments, related work.'
    ],
    llmRequired: true,
    codeLink: 'https://gitfront.io/r/anonymous-repo-acl/ecCqeCKyx8tM/ReviewGrounder-ACL-26-submission/'
  }
];
