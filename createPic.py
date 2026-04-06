import base64

def generate_svg_from_rsdpl_workflow():
    # Defines the entire SVG image as a raw string
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1300" font-family="Arial, Helvetica, sans-serif" font-size="14">
  <defs>
    <linearGradient id="mllm_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#dfebf5;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#cfe2f3;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="opt_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f8e6eb;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#f1c8d8;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="gen_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#e8f4eb;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#cfe8d3;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="text_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#fafafa;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#f0f0f0;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="vis_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#fcfaf2;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#fcf1c7;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="know_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#e6f3f0;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#cfe6e1;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="result_grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f0f8ff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#e0f0ff;stop-opacity:1" />
    </linearGradient>

    <marker id="arrow_head" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333" />
    </marker>
    <marker id="arrow_head_dashed" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#666" />
    </marker>

    <filter id="shadow">
      <feDropShadow dx="1" dy="1" stdDeviation="1.5" flood-color="#aaa" flood-opacity="0.5"/>
    </filter>
  </defs>

  <rect x="0" y="0" width="1000" height="1300" fill="#fff" />
  <text x="500" y="30" text-anchor="middle" font-size="24" font-weight="bold">RS-DPL: Retrieval-augmented and Self-prompted Discrete Prompt Learning</text>
  <text x="500" y="55" text-anchor="middle" font-size="16" fill="#666">(基于检索增强生成与自我提示的离散提示学习完整执行流水线)</text>

  <g id="step1_elements" transform="translate(50, 80)">
    <text x="-30" y="20" font-weight="bold" font-size="18">1.</text>
    <rect x="0" y="0" width="300" height="40" rx="5" fill="#fdfdfd" stroke="#bbb"/>
    <text x="150" y="25" text-anchor="middle" font-weight="bold">基础前向评估与错误样本筛选</text>
    <text x="150" y="45" text-anchor="middle" font-size="12">(Initial Eval &amp; Error Sample Filtering)</text>
    
    <g transform="translate(0, 70)">
      <rect x="0" y="0" width="120" height="80" rx="5" fill="url(#text_grad)" stroke="#bbb" filter="url(#shadow)"/>
      <text x="60" y="25" text-anchor="middle">原始输入</text>
      <text x="60" y="45" text-anchor="middle" font-size="12">($I_{orig}, Q_{text}$)</text>
      <text x="60" y="65" text-anchor="middle" font-size="12">True Label: $Y_{true}$</text>
      
      <rect x="180" y="0" width="120" height="80" rx="10" fill="url(#mllm_grad)" stroke="#bbb" filter="url(#shadow)"/>
      <text x="240" y="25" text-anchor="middle">基础多模态</text>
      <text x="240" y="45" text-anchor="middle">大模型</text>
      <text x="240" y="65" text-anchor="middle" font-size="12">(Base MLLM, $f_\theta$)</text>
      
      <rect x="360" y="20" width="80" height="40" rx="5" fill="#fdfdfd" stroke="#bbb" filter="url(#shadow)"/>
      <text x="400" y="45" text-anchor="middle">预测标签</text>
      <text x="400" y="65" text-anchor="middle" font-size="12">($\hat{Y}$)</text>
      
      <path d="M 120 40 L 180 40" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
      <path d="M 300 40 L 360 40" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
    </g>
    
    <g transform="translate(480, 70)">
        <polygon points="0,40 40,0 80,40 40,80" fill="#fcfcfc" stroke="#333"/>
        <text x="40" y="45" text-anchor="middle" font-size="20">?</text>
        <text x="40" y="65" text-anchor="middle" font-size="12">$\hat{Y} \neq Y_{true}$</text>
        
        <path d="M -40 40 L 0 40" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
        
        <rect x="140" y="10" width="180" height="60" rx="5" fill="url(#text_grad)" stroke="#e44c4c" stroke-dasharray="5,5" filter="url(#shadow)"/>
        <text x="230" y="30" text-anchor="middle" font-weight="bold">错误样本集合</text>
        <text x="230" y="50" text-anchor="middle" font-size="12">($\mathcal{D}_{error}$ = {($I_{orig}, Q_{text}, \hat{Y}, Y_{true}$)})</text>
        
        <path d="M 80 40 L 140 40" stroke="#e44c4c" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrow_head_dashed)"/>
        <text x="110" y="35" text-anchor="middle" font-size="12" fill="#e44c4c">Yes</text>
    </g>
  </g>

  <g id="step23_row" transform="translate(50, 310)">
    <g id="step2_elements">
      <text x="-30" y="20" font-weight="bold" font-size="18">2.</text>
      <rect x="0" y="0" width="300" height="40" rx="5" fill="#fdfdfd" stroke="#bbb"/>
      <text x="150" y="25" text-anchor="middle" font-weight="bold">基于预定义子问题的图转文解析</text>
      <text x="150" y="45" text-anchor="middle" font-size="12">(Image-to-Text Parsing, Self-prompting)</text>
      
      <g transform="translate(0, 70)">
        <rect x="180" y="10" width="120" height="80" rx="10" fill="url(#opt_grad)" stroke="#bbb" filter="url(#shadow)"/>
        <text x="240" y="35" text-anchor="middle" font-weight="bold">优化器模型</text>
        <text x="240" y="55" text-anchor="middle" font-size="12">(Optimizer Model, $M_{opt}$)</text>
        <text x="240" y="75" text-anchor="middle" font-size="12">回答子问题</text>
        
        <g transform="translate(240, 10) scale(0.6)">
          <rect x="-80" y="20" width="40" height="40" rx="2" fill="#fcfcfc" stroke="#bbb"/>
          <text x="-60" y="45" text-anchor="middle" font-size="16">Obj</text>
          
          <rect x="-20" y="20" width="40" height="40" rx="2" fill="#fcfcfc" stroke="#bbb"/>
          <text x="0" y="45" text-anchor="middle" font-size="16">App</text>
          
          <rect x="-80" y="70" width="40" height="40" rx="2" fill="#fcfcfc" stroke="#bbb"/>
          <text x="-60" y="95" text-anchor="middle" font-size="16">Env</text>
          
          <rect x="-20" y="70" width="40" height="40" rx="2" fill="#fcfcfc" stroke="#bbb"/>
          <text x="0" y="95" text-anchor="middle" font-size="16">Emo</text>
        </g>
        
        <path d="M 0 50 L 180 50" stroke="#666" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrow_head_dashed)"/>
        <text x="90" y="45" text-anchor="middle" font-size="12" fill="#666">(from $\mathcal{D}_{error}$)</text>
        
        <rect x="360" y="10" width="120" height="80" rx="5" fill="url(#text_grad)" stroke="#bbb" filter="url(#shadow)"/>
        <text x="420" y="35" text-anchor="middle" font-weight="bold">纯文本描述</text>
        <text x="420" y="55" text-anchor="middle" font-size="12">($D_{parsed}$)</text>
        <text x="420" y="75" text-anchor="middle" font-size="10">(Objects, Appearance, Environment, Emotions)</text>
        
        <path d="M 300 50 L 360 50" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
      </g>
    </g>
    
    <line x1="510" y1="0" x2="510" y2="170" stroke="#ccc" stroke-width="2" stroke-dasharray="8,8"/>
    
    <g id="step3_elements" transform="translate(540, 0)">
      <text x="-30" y="20" font-weight="bold" font-size="18">3.</text>
      <rect x="0" y="0" width="300" height="40" rx="5" fill="#fdfdfd" stroke="#bbb"/>
      <text x="150" y="25" text-anchor="middle" font-weight="bold">基于文本描述的纯文本RAG检索</text>
      <text x="150" y="45" text-anchor="middle" font-size="12">(Text-only RAG Retrieval)</text>
      
      <g transform="translate(0, 70)">
        <rect x="0" y="10" width="120" height="80" rx="5" fill="#fdfdfd" stroke="#bbb" filter="url(#shadow)"/>
        <text x="60" y="35" text-anchor="middle" font-weight="bold">查询构建</text>
        <text x="60" y="55" text-anchor="middle" font-size="12">拼接 ($Q_{text}, D_{parsed}$)</text>
        <text x="60" y="75" text-anchor="middle" font-size="12">($Q_{RAG}$)</text>
        
        <path d="M -60 50 C -40 50, -40 50, -20 50" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)" />
        <text x="-40" y="45" text-anchor="middle" font-size="12" fill="#333">($D_{parsed}$)</text>
        <path d="M -60 20 C -40 20, -40 20, 0 50" stroke="#666" stroke-width="1.5" stroke-dasharray="5,5" marker-end="url(#arrow_head_dashed)" />
        <text x="-30" y="15" text-anchor="middle" font-size="12" fill="#666">($Q_{text}$)</text>
        
        <g transform="translate(180, 10)">
            <rect x="0" y="0" width="120" height="80" rx="5" fill="url(#know_grad)" stroke="#bbb" filter="url(#shadow)"/>
            <text x="60" y="30" text-anchor="middle" font-weight="bold">外部知识库</text>
            <text x="60" y="50" text-anchor="middle" font-size="12">($K_{ext}$)</text>
            <ellipse cx="60" cy="65" rx="15" ry="5" fill="#fcfcfc" stroke="#bbb"/>
            <rect x="45" y="65" width="30" height="10" fill="#fcfcfc" stroke="none"/>
            <path d="M 45 65 L 45 75 A 15 5 0 0 0 75 75 L 75 65" fill="#fcfcfc" stroke="#bbb"/>
        </g>
        
        <path d="M 120 50 L 180 50" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
        <text x="150" y="45" text-anchor="middle" font-size="12">检索</text>
        
        <rect x="360" y="10" width="120" height="80" rx="5" fill="url(#text_grad)" stroke="#bbb" filter="url(#shadow)"/>
        <text x="420" y="35" text-anchor="middle" font-weight="bold">召回知识</text>
        <text x="420" y="55" text-anchor="middle" font-size="12">($K_{retrived}$)</text>
        <text x="420" y="75" text-anchor="middle" font-size="12">(客观辅助知识)</text>
        
        <path d="M 300 50 L 360 50" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
      </g>
    </g>
  </g>

  <g id="step4_elements" transform="translate(50, 560)">
    <text x="-30" y="20" font-weight="bold" font-size="18">4.</text>
    <rect x="0" y="0" width="900" height="40" rx="5" fill="#fdfdfd" stroke="#bbb"/>
    <text x="450" y="25" text-anchor="middle" font-weight="bold">双层元学习提示词优化</text>
    <text x="450" y="45" text-anchor="middle" font-size="12">(Two-level Meta-learning Prompt Optimization)</text>
    
    <rect x="0" y="60" width="900" height="380" rx="10" fill="#fcfcfc" stroke="#666" stroke-dasharray="10,10"/>
    <text x="450" y="80" text-anchor="middle" font-weight="bold" font-size="16">内层循环 (局部子任务，联合优化)</text>
    <text x="450" y="100" text-anchor="middle" font-size="12">(Inner Loop: Local Joint Optimization)</text>
    
    <g transform="translate(150, 120)">
        <rect x="0" y="0" width="120" height="50" rx="5" fill="url(#text_grad)" stroke="#bbb" filter="url(#shadow)"/>
        <text x="60" y="20" text-anchor="middle">固定</text>
        <text x="60" y="40" text-anchor="middle" font-weight="bold">系统提示词</text>
        <text x="60" y="60" text-anchor="middle" font-size="12">($P_{sys}$)</text>
        
        <g transform="translate(180, 0)">
            <rect x="0" y="0" width="240" height="120" rx="10" fill="url(#opt_grad)" stroke="#bbb" filter="url(#shadow)"/>
            <text x="120" y="30" text-anchor="middle" font-weight="bold" font-size="16">优化器模型 ($M_{opt}$)</text>
            <rect x="20" y="50" width="200" height="50" rx="5" fill="#fcfcfc" stroke="#bbb"/>
            <text x="120" y="70" text-anchor="middle">综合错误原因分析</text>
            <text x="120" y="90" text-anchor="middle" font-size="12">(Comprehensive Error Analysis)</text>
            
            <path d="M -60 25 L 0 25" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
            <path d="M -60 180 L 0 100" stroke="#666" stroke-width="1.5" stroke-dasharray="5,5" marker-end="url(#arrow_head_dashed)"/>
            <text x="-30" y="150" text-anchor="middle" font-size="12" fill="#666">(from previous steps)</text>
        </g>
        
        <g transform="translate(480, 0)">
            <rect x="0" y="0" width="200" height="100" rx="10" fill="url(#gen_grad)" stroke="#bbb" filter="url(#shadow)"/>
            <text x="100" y="35" text-anchor="middle" font-weight="bold">图像生成器</text>
            <text x="100" y="55" text-anchor="middle" font-size="12">(Image Generator, e.g., GPT-Image)</text>
            <text x="100" y="75" text-anchor="middle" font-size="12">生成/编辑/混合</text>
            
            <path d="M -60 60 L 0 60" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
            <text x="-30" y="55" text-anchor="middle" font-size="12">文本条件 ($C_{text}$)</text>
        </g>
        
        <rect x="480" y="140" width="200" height="60" rx="5" fill="url(#vis_grad)" stroke="#bbb" filter="url(#shadow)"/>
        <text x="580" y="160" text-anchor="middle" font-weight="bold">视觉提示词</text>
        <text x="580" y="180" text-anchor="middle" font-size="12">(视觉参考图, $P_{vis}^{(new)}$)</text>
        
        <path d="M 580 100 L 580 140" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
        
        <g transform="translate(100, 240)">
            <rect x="0" y="0" width="400" height="100" rx="10" fill="url(#mllm_grad)" stroke="#bbb" filter="url(#shadow)"/>
            <text x="200" y="35" text-anchor="middle" font-weight="bold" font-size="16">组合寻优 (Base MLLM Evaluation)</text>
            <text x="200" y="55" text-anchor="middle" font-size="12">(Combinatorial Optimization)</text>
            
            <rect x="40" y="65" width="320" height="25" rx="5" fill="#fcfcfc" stroke="#bbb"/>
            <text x="200" y="82" text-anchor="middle">性能最大化: $ \max_{\{(P_{text}^{(cand)}, P_{vis}^{(cand)}) \}} \text{Perf}(f_\theta(I_{orig}, P_{sys}, P_{text}, P_{vis})) $</text>
            
            <path d="M 400 170 L 400 240" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
            
            <path d="M 200 120 L 200 240" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)" />
            <text x="200" y="160" text-anchor="middle" font-size="12">候选文本用户提示 ($P_{text\_user}^{(new)}$)</text>
            
            <path d="M 0 50 L -100 50" stroke="#e44c4c" stroke-width="2" marker-end="url(#arrow_head)"/>
            <text x="-50" y="45" text-anchor="middle" font-size="12" fill="#e44c4c">局部更新</text>
            <rect x="-220" y="20" width="120" height="60" rx="5" fill="url(#result_grad)" stroke="#e44c4c" filter="url(#shadow)"/>
            <text x="-160" y="40" text-anchor="middle">完成局部更新</text>
            <text x="-160" y="60" text-anchor="middle" font-weight="bold">($P_{vis}^*, P_{text\_user}^*$)</text>
        </g>
    </g>
    
    <path d="M 450 440 L 450 480" stroke="#333" stroke-width="3" stroke-dasharray="10,10" marker-end="url(#arrow_head)"/>
    <text x="450" y="470" text-anchor="middle" font-size="12" fill="#666">(内层收敛后触发)</text>
    
    <rect x="0" y="480" width="900" height="230" rx="10" fill="#fefefe" stroke="#e44c4c" stroke-width="2" filter="url(#shadow)"/>
    <text x="450" y="500" text-anchor="middle" font-weight="bold" font-size="16">外层循环 (系统提示词优化，全局跨任务)</text>
    <text x="450" y="520" text-anchor="middle" font-size="12">(Outer Loop: Global System Prompt Optimization)</text>
    
    <g transform="translate(50, 540)">
        <rect x="0" y="0" width="160" height="60" rx="5" fill="url(#know_grad)" stroke="#bbb" filter="url(#shadow)"/>
        <text x="80" y="25" text-anchor="middle">聚合所有源任务</text>
        <text x="80" y="45" text-anchor="middle">综合分析结果</text>
        
        <g transform="translate(80, 60) scale(0.6)">
            <rect x="-80" y="10" width="30" height="30" rx="2" fill="#fcfcfc" stroke="#bbb"/>
            <text x="-65" y="32" text-anchor="middle" font-size="14">A</text>
            <rect x="-30" y="10" width="30" height="30" rx="2" fill="#fcfcfc" stroke="#bbb"/>
            <text x="-15" y="32" text-anchor="middle" font-size="14">B</text>
            <rect x="20" y="10" width="30" height="30" rx="2" fill="#fcfcfc" stroke="#bbb"/>
            <text x="35" y="32" text-anchor="middle" font-size="14">C</text>
            <text x="65" y="30" font-size="16">...</text>
        </g>
        
        <g transform="translate(200, 0)">
            <rect x="0" y="0" width="240" height="110" rx="10" fill="url(#opt_grad)" stroke="#bbb" filter="url(#shadow)"/>
            <text x="120" y="30" text-anchor="middle" font-weight="bold" font-size="16">优化器模型 ($M_{opt}$)</text>
            <rect x="20" y="50" width="200" height="50" rx="5" fill="#fcfcfc" stroke="#bbb"/>
            <text x="120" y="70" text-anchor="middle">跨任务共性错误分析</text>
            <text x="120" y="90" text-anchor="middle" font-size="12">(Cross-task Error Analysis)</text>
            
            <path d="M -40 30 L 0 30" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
            
            <path d="M 240 55 L 280 55" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
            <text x="260" y="45" text-anchor="middle" font-size="12">迭代</text>
        </g>
        
        <rect x="480" y="25" width="160" height="60" rx="5" fill="url(#text_grad)" stroke="#bbb" filter="url(#shadow)"/>
        <text x="560" y="45" text-anchor="middle">候选全局</text>
        <text x="560" y="65" text-anchor="middle" font-weight="bold">系统提示词</text>
        <text x="560" y="85" text-anchor="middle" font-size="12">($P_{sys}^{(cand)}$)</text>
        
        <path d="M 440 55 L 480 55" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
        
        <g transform="translate(680, 0)">
            <polygon points="0,55 40,15 80,55 40,95" fill="#fcfcfc" stroke="#333"/>
            <text x="40" y="60" text-anchor="middle" font-size="12">测试 &amp;</text>
            <text x="40" y="75" text-anchor="middle" font-size="12">筛选</text>
            
            <path d="M -40 55 L 0 55" stroke="#333" stroke-width="2" marker-end="url(#arrow_head)"/>
            <text x="-20" y="45" text-anchor="middle" font-size="12">测试泛化力</text>
            
            <path d="M 80 55 L 120 55" stroke="#333" stroke-width="3" marker-end="url(#arrow_head)"/>
            <text x="100" y="45" text-anchor="middle" font-size="12" fill="#333">最优</text>
            
            <rect x="120" y="25" width="160" height="60" rx="5" fill="url(#result_grad)" stroke="#e44c4c" stroke-width="2" filter="url(#shadow)"/>
            <text x="200" y="45" text-anchor="middle">最终最优全局</text>
            <text x="200" y="65" text-anchor="middle" font-weight="bold">系统提示词</text>
            <text x="200" y="85" text-anchor="middle" font-size="12">($P_{sys}^*$)</text>
        </g>
    </g>
  </g>

  <g id="summary_section" transform="translate(50, 1150)">
    <rect x="0" y="0" width="900" height="120" rx="10" fill="#f0f8ff" stroke="#e0e0e0" />
    <text x="450" y="25" text-anchor="middle" font-size="20" font-weight="bold">RS-DPL 最终优化提示词构成</text>
    
    <g transform="translate(100, 40)">
        <rect x="0" y="0" width="200" height="70" rx="5" fill="url(#text_grad)" stroke="#e44c4c" stroke-width="2"/>
        <text x="100" y="30" text-anchor="middle" font-size="16" font-weight="bold">全局系统提示词</text>
        <text x="100" y="55" text-anchor="middle" font-size="14">($P_{sys}^*$)</text>
    </g>
    
    <g transform="translate(350, 40)">
        <rect x="0" y="0" width="200" height="70" rx="5" fill="url(#text_grad)" stroke="#bbb"/>
        <text x="100" y="30" text-anchor="middle" font-size="16" font-weight="bold">特定用户提示</text>
        <text x="100" y="55" text-anchor="middle" font-size="14">($P_{text\_user}^*$)</text>
    </g>
    
    <g transform="translate(600, 40)">
        <rect x="0" y="0" width="200" height="70" rx="5" fill="url(#vis_grad)" stroke="#bbb"/>
        <text x="100" y="30" text-anchor="middle" font-size="16" font-weight="bold">特定视觉提示</text>
        <text x="100" y="55" text-anchor="middle" font-size="14">($P_{vis}^*$)</text>
    </g>
  </g>

</svg>"""
    return base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')

# The SVG is generated as a base64 string for display or download
rsdpl_workflow_svg_b64 = generate_svg_from_rsdpl_workflow()
# This base64 string can be used as the source of an <img> tag in HTML:
# <img src="data:image/svg+xml;base64,..." alt="RS-DPL Workflow" />

# ... 在原有代码末尾添加 ...
if __name__ == "__main__":
    # 获取 base64 字符串
    b64_str = generate_svg_from_rsdpl_workflow()
    
    # 解码并写入文件
    
    svg_data = base64.b64decode(b64_str)
    
    with open("rs_dpl_framework.svg", "wb") as f:
        f.write(svg_data)
        
    print("SVG 框架图已生成：rs_dpl_framework.svg")