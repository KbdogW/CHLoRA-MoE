<template>
  <div class="min-h-screen p-8 bg-gray-50">
    <div class="max-w-6xl mx-auto space-y-6">
      <!-- Header -->
      <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-800">语音病理辅助诊断系统</h1>
          <p class="text-gray-500 mt-1 text-sm">基于 WavLM MoE 的高并发诊断客户端</p>
        </div>
        <el-tag :type="serverStatus === 'connected' ? 'success' : 'danger'" effect="light" round>
          {{ serverStatus === 'connected' ? '后端服务已连接' : '后端服务未连接' }}
        </el-tag>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- 左侧：数据输入区 -->
        <div class="lg:col-span-4 space-y-6">
          <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-full flex flex-col">
            <h2 class="text-lg font-semibold mb-4 text-gray-700 flex items-center">
              <el-icon class="mr-2"><Microphone /></el-icon> 音频采集
            </h2>
            
            <el-tabs v-model="activeTab" class="flex-grow flex flex-col">
              <el-tab-pane label="本地上传" name="upload">
                <el-upload
                  class="mt-4"
                  drag
                  action="#"
                  :auto-upload="false"
                  :on-change="handleFileChange"
                  accept="audio/wav"
                  :disabled="isLoading"
                >
                  <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                  <div class="el-upload__text">
                    拖拽 .wav 文件到此处，或 <em>点击上传</em>
                  </div>
                  <template #tip>
                    <div class="el-upload__tip text-center">
                      仅支持 16kHz .wav 格式文件
                    </div>
                  </template>
                </el-upload>
                
                <el-button 
                  type="primary" 
                  class="w-full mt-6" 
                  size="large"
                  :loading="isLoading"
                  :disabled="!selectedFile"
                  @click="submitAudio(selectedFile)"
                >
                  开始智能分析
                </el-button>
              </el-tab-pane>
              
              <el-tab-pane label="实时录音" name="record">
                <div class="flex flex-col items-center justify-center py-10">
                  <div 
                    class="w-32 h-32 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300 shadow-md"
                    :class="isRecording ? 'bg-red-50 hover:bg-red-100 border-4 border-red-500 animate-pulse' : 'bg-blue-50 hover:bg-blue-100 border border-blue-200'"
                    @click="toggleRecording"
                  >
                    <el-icon :size="48" :color="isRecording ? '#f56c6c' : '#409eff'">
                      <Microphone />
                    </el-icon>
                  </div>
                  <p class="mt-6 text-gray-500 font-medium">
                    {{ isRecording ? '录音中... 点击停止' : '点击开始录音' }}
                  </p>
                </div>
              </el-tab-pane>
            </el-tabs>
          </div>
        </div>

        <!-- 右侧：诊断结果区 -->
        <div class="lg:col-span-8">
          <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100 min-h-[600px] relative">
            <h2 class="text-lg font-semibold mb-4 text-gray-700 flex items-center">
              <el-icon class="mr-2"><DataAnalysis /></el-icon> 诊断报告
            </h2>

            <!-- 加载状态 -->
            <div v-if="isLoading" class="absolute inset-0 z-10 bg-white/80 flex flex-col items-center justify-center rounded-xl backdrop-blur-sm">
              <el-icon class="is-loading text-blue-500 mb-4" :size="48"><Loading /></el-icon>
              <p class="text-gray-600 font-medium">Triton 模型推理中，请稍候...</p>
            </div>

            <!-- 空状态 -->
            <el-empty v-else-if="!result" description="暂无诊断数据，请在左侧上传音频或录音" />

            <!-- 结果展示 -->
            <div v-else class="space-y-6 fade-in">
              <!-- 核心指标卡片 -->
              <div class="grid grid-cols-2 gap-4">
                <div class="bg-blue-50 rounded-lg p-5 border border-blue-100">
                  <p class="text-sm text-blue-600 mb-1">评估结果 (Severity)</p>
                  <p class="text-3xl font-bold text-gray-800">{{ result.prediction.severity_label }}</p>
                </div>
                <div class="bg-green-50 rounded-lg p-5 border border-green-100">
                  <p class="text-sm text-green-600 mb-1">置信度 (Confidence)</p>
                  <p class="text-3xl font-bold text-gray-800">{{ (result.prediction.confidence * 100).toFixed(1) }}%</p>
                </div>
              </div>

              <!-- 诊断意见 -->
              <div class="bg-gray-50 p-5 rounded-lg border border-gray-200">
                <p class="text-sm font-semibold text-gray-600 mb-2">基础临床诊断：</p>
                <p class="text-gray-800 leading-relaxed">{{ result.prediction.diagnosis_report }}</p>
                <div class="mt-3 text-xs text-gray-400 flex justify-between">
                  <span>文件 ID: {{ result.file_id }}</span>
                  <span>推理耗时: {{ (result.timing ? result.timing.triton_inference_sec : result.inference_time_sec) * 1000 }} ms</span>
                  <span>生成耗时: {{ result.timing ? (result.timing.rag_generation_sec * 1000).toFixed(1) : 0 }} ms</span>
                  <span>路由决策: {{ result.details.router_decision }}</span>
                </div>
              </div>

              <!-- AI 专家 RAG 报告 -->
              <div v-if="result.rag_clinical_advice" class="bg-blue-50/50 p-6 rounded-xl border border-blue-100 shadow-sm relative overflow-hidden">
                <div class="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                <h3 class="text-lg font-bold text-blue-800 mb-4 flex items-center">
                  <el-icon class="mr-2"><Document /></el-icon> AI 专家诊断与康复建议 (基于知识库增强)
                </h3>
                <div class="prose prose-blue max-w-none text-gray-700 leading-relaxed text-sm" v-html="parsedRagReport"></div>
              </div>

              <!-- 可视化图表 -->
              <div class="grid grid-cols-2 gap-6 mt-6">
                <div class="border border-gray-100 rounded-lg p-4 shadow-sm">
                  <h3 class="text-sm font-medium text-gray-600 mb-2 text-center">严重程度概率分布</h3>
                  <div id="probChart" class="h-64 w-full" style="height: 300px; width: 100%;"></div>
                </div>
                <div class="border border-gray-100 rounded-lg p-4 shadow-sm">
                  <h3 class="text-sm font-medium text-gray-600 mb-2 text-center">MoE 专家路由权重</h3>
                  <div id="routerChart" class="h-64 w-full" style="height: 300px; width: 100%;"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, computed } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'
import * as echarts from 'echarts'
import { marked } from 'marked'

// 后端 API 地址，如果你在本地浏览器访问，请把 10.10.10.113 换成你服务器的实际局域网 IP
const API_BASE_URL = 'http://10.10.10.113:8080'

// 状态变量
const activeTab = ref('upload')
const serverStatus = ref('disconnected')
const isLoading = ref(false)
const selectedFile = ref<File | null>(null)
const result = ref<any>(null)

// 转换 RAG 报告为 Markdown
const parsedRagReport = computed(() => {
  if (result.value && result.value.rag_clinical_advice) {
    return marked.parse(result.value.rag_clinical_advice)
  }
  return ''
})

// ECharts 实例引用，用于在组件卸载时清理
let probChartInstance: echarts.ECharts | null = null
let routerChartInstance: echarts.ECharts | null = null

// 录音相关变量
const isRecording = ref(false)
let mediaRecorder: MediaRecorder | null = null
let audioChunks: Blob[] = []

// 初始化检查后端状态
onMounted(async () => {
  try {
    await axios.get(`${API_BASE_URL}/health`)
    serverStatus.value = 'connected'
  } catch (e) {
    serverStatus.value = 'disconnected'
    ElMessage.warning('未能连接到后端服务，请确认 fastapi 是否启动。')
  }
})

// 处理文件选择
const handleFileChange = (uploadFile: any) => {
  selectedFile.value = uploadFile.raw
}

// 录音逻辑控制
const toggleRecording = async () => {
  if (isRecording.value) {
    stopRecording()
  } else {
    startRecording()
  }
}

const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 })
    const source = audioContext.createMediaStreamSource(stream)
    const processor = audioContext.createScriptProcessor(4096, 1, 1)

    audioChunks = [] // 这里我们将存储重采样后的 Float32 音频数据

    processor.onaudioprocess = (e) => {
      if (!isRecording.value) return
      const inputData = e.inputBuffer.getChannelData(0)
      // 复制 Float32Array 数据
      audioChunks.push(new Float32Array(inputData))
    }

    source.connect(processor)
    processor.connect(audioContext.destination)
    
    // 保存引用以便后续清理
    ;(window as any).currentAudioContext = audioContext
    ;(window as any).currentStream = stream
    ;(window as any).currentProcessor = processor
    ;(window as any).currentSource = source

    isRecording.value = true
  } catch (err) {
    ElMessage.error('无法访问麦克风，请检查浏览器权限 (可能需要 HTTPS)')
    console.error(err)
  }
}

const stopRecording = () => {
  if (isRecording.value) {
    isRecording.value = false
    
    const audioContext = (window as any).currentAudioContext
    const stream = (window as any).currentStream
    const processor = (window as any).currentProcessor
    const source = (window as any).currentSource

    if (processor && source && audioContext) {
      source.disconnect()
      processor.disconnect()
      audioContext.close()
    }

    if (stream) {
      stream.getTracks().forEach((track: MediaStreamTrack) => track.stop())
    }

    // 将所有块合并为单个 Float32Array
    let totalLength = 0
    audioChunks.forEach((chunk: any) => totalLength += chunk.length)
    const mergedData = new Float32Array(totalLength)
    let offset = 0
    audioChunks.forEach((chunk: any) => {
      mergedData.set(chunk, offset)
      offset += chunk.length
    })

    // 将 Float32 PCM 数据转换为 16-bit PCM WAV 格式
    const wavBlob = encodeWAV(mergedData, 16000)
    const file = new File([wavBlob], "recorded_audio.wav", { type: 'audio/wav' })
    
    submitAudio(file) // 提交标准格式的音频
  }
}

// 将 Float32Array 转换为标准的 16-bit PCM WAV 格式
function encodeWAV(samples: Float32Array, sampleRate: number) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i))
    }
  }

  /* RIFF identifier */
  writeString(view, 0, 'RIFF')
  /* RIFF chunk length */
  view.setUint32(4, 36 + samples.length * 2, true)
  /* RIFF type */
  writeString(view, 8, 'WAVE')
  /* format chunk identifier */
  writeString(view, 12, 'fmt ')
  /* format chunk length */
  view.setUint32(16, 16, true)
  /* sample format (raw) */
  view.setUint16(20, 1, true)
  /* channel count (1=mono) */
  view.setUint16(22, 1, true)
  /* sample rate */
  view.setUint32(24, sampleRate, true)
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true)
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true)
  /* bits per sample */
  view.setUint16(34, 16, true)
  /* data chunk identifier */
  writeString(view, 36, 'data')
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true)

  /* 写入 16-bit PCM 采样数据 */
  let offset = 44
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
  }

  return new Blob([view], { type: 'audio/wav' })
}

// 提交音频文件到后端
const submitAudio = async (file: File | null) => {
  if (!file) {
    ElMessage.warning('请先提供音频文件')
    return
  }

  isLoading.value = true
  result.value = null
  
  // 销毁旧的图表实例
  if (probChartInstance) {
    probChartInstance.dispose()
    probChartInstance = null
  }
  if (routerChartInstance) {
    routerChartInstance.dispose()
    routerChartInstance = null
  }

  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    if (response.data.status === 'success') {
      result.value = response.data
      ElMessage.success('诊断完成！')
      
      // 使用 setTimeout 配合 nextTick，确保含有 Tailwind 类名的 DOM 完全就绪 (具有实际宽高) 后再初始化 ECharts
      nextTick(() => {
        setTimeout(() => {
           renderCharts()
        }, 150) // 给予 150ms 延迟让浏览器完成排版渲染
      })
    }
  } catch (error: any) {
    console.error(error)
    ElMessage.error(error.response?.data?.detail || '请求失败，请检查网络或后端状态')
  } finally {
    isLoading.value = false
  }
}

// 渲染 ECharts 图表
const renderCharts = () => {
  if (!result.value) return

  const { class_probabilities, router_probabilities } = result.value.details

  // 1. 严重程度概率柱状图
  const probChartDOM = document.getElementById('probChart')
  
  // 安全检查：如果 DOM 不存在，再等一会儿重试
  if (!probChartDOM) {
      setTimeout(renderCharts, 50)
      return
  }

  probChartInstance = echarts.init(probChartDOM)
  probChartInstance.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: Object.keys(class_probabilities) },
    yAxis: { type: 'value', max: 1 },
    series: [
      {
        data: Object.values(class_probabilities),
        type: 'bar',
        itemStyle: { color: '#409eff', borderRadius: [4, 4, 0, 0] },
        barWidth: '50%'
      }
    ]
  })

  // 2. 路由权重饼图
  const routerChartDOM = document.getElementById('routerChart')
  if (routerChartDOM) {
    routerChartInstance = echarts.init(routerChartDOM)
    const pieData = Object.entries(router_probabilities).map(([name, value]) => ({
      name, value
    }))
    
    routerChartInstance.setOption({
      tooltip: { trigger: 'item' },
      series: [
        {
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: { show: false, position: 'center' },
          emphasis: {
            label: { show: true, fontSize: 16, fontWeight: 'bold' }
          },
          labelLine: { show: false },
          data: pieData
        }
      ]
    })
  }

  // 监听窗口大小变化，让图表自适应缩放
  window.addEventListener('resize', () => {
    if (probChartInstance) probChartInstance.resize()
    if (routerChartInstance) routerChartInstance.resize()
  })
}
</script>

<style scoped>
.fade-in {
  animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
