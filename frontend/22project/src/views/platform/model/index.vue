<template>
	<!-- 必须包一层单根 div：本页除了 el-tabs 还有 3 个 el-dialog，
	     多根 fragment 会让框架的 KeepAlive/Transition 报
	     "Extraneous non-props attributes / non-element root node" 且 class 挂不上 -->
	<div class="platform-page">
	<el-tabs v-model="tab">
		<!-- ============ 模型清单 ============ -->
		<el-tab-pane label="模型清单" name="list">
			<el-card shadow="never">
				<template #header>
					<span>模型清单（登记信息 + 模型参数）</span>
					<span class="hint" style="margin-left: 8px">点任意一行，下方展示该模型的完整参数</span>
					<span style="float: right">
						<el-button type="primary" size="small" @click="openUpload">上传模型</el-button>
						<el-button size="small" @click="loadModels">刷新</el-button>
					</span>
				</template>
				<el-table :data="mergedRows" size="small" highlight-current-row empty-text="还没有登记/产物"
					@current-change="onRowClick" style="cursor: pointer">
					<!-- 四栏都用 min-width 且取值相同：el-table 会按 min-width 比例分配剩余宽度，
					     于是四栏最终等宽（原来是"头栏吃掉全部余量"，看着忽宽忽窄） -->
					<el-table-column label="模型 / 登记信息" min-width="160">
						<template #default="{ row }">
							<b>{{ row.name }}</b>
							<el-tag v-if="row.reg?.ModelType" size="small" class="ml">{{ typeLabel(row.reg.ModelType) }}</el-tag>
							<div class="hint">ID={{ row.reg?.ModelID ?? '—' }} · 接口 {{ row.reg?.ApiEndpoint || '—' }}</div>
							<div class="hint">状态 {{ row.reg?.Status || '—' }} · {{ row.reg?.IsActive ? '已启用' : '已停用' }}</div>
						</template>
					</el-table-column>
					<!-- 列表只回答「有没有产物、什么框架、第几版」；参数细节点行看下方详情卡，不重复搬运 -->
					<el-table-column label="产物" min-width="160" align="center">
						<template #default="{ row }">
							<template v-if="row.art">
								<el-tag size="small" type="success">{{ fwLabel(row.art.framework) }}</el-tag>
								<span class="ml">{{ row.art.version }}</span>
							</template>
							<span v-else class="hint">无产物</span>
						</template>
					</el-table-column>
					<el-table-column label="指标" min-width="160" align="center">
						<template #default="{ row }">
							<template v-if="row.art">
								<b v-if="rowMetric(row).value !== '—'">{{ rowMetric(row).value }}</b>
								<span v-else class="hint">—</span>
								<div class="hint">{{ rowMetric(row).label }}</div>
							</template>
							<span v-else class="hint">无产物</span>
						</template>
					</el-table-column>
					<el-table-column label="操作" min-width="160" align="center">
						<template #default="{ row }">
							<el-button link type="primary" @click.stop="openEdit(row)">编辑</el-button>
							<el-button link type="danger" @click.stop="removeModel(row)">删除</el-button>
						</template>
					</el-table-column>
				</el-table>
			</el-card>

			<!-- ====== 列表下方：所选模型的完整参数 ====== -->
			<el-card shadow="never" class="mt">
				<template #header>
					<span>模型展示：{{ overview?.model || '未选择' }} 的模型参数</span>
					<span class="hint" style="margin-left: 8px">点上方列表任意一行查看</span>
					<el-button v-if="overview" link type="primary" style="float: right" @click="showMeta(overview.model)">meta.json 全文</el-button>
				</template>
				<div v-if="!overview" class="empty">尚未选择模型</div>
				<template v-else>
					<el-descriptions :column="3" border size="small">
						<el-descriptions-item label="框架">{{ latestArt.framework || '—' }}</el-descriptions-item>
						<el-descriptions-item label="版本">{{ overview.latest_version || '—' }}</el-descriptions-item>
						<el-descriptions-item label="输入长度">{{ latestArt.input_len ?? '—' }}</el-descriptions-item>

						<!-- 分类/回归看"类别数"，异常检测没有类别，改看检测器与判定阈值 -->
						<el-descriptions-item v-if="!isAnomalyModel" label="类别数">
							{{ latestArt.num_classes ?? '—' }}
						</el-descriptions-item>
						<template v-else>
							<el-descriptions-item label="检测器">
								{{ overview.params?.detector || '—' }}<span v-if="overview.params?.k != null" class="hint"> k={{ overview.params.k }}</span>
							</el-descriptions-item>
							<el-descriptions-item label="判定阈值">{{ fmtThreshold(overview.metrics?.threshold) }}</el-descriptions-item>
						</template>

						<el-descriptions-item label="权重文件">
							{{ (latestArt.weights || '').split('\\').pop() || '—' }}
						</el-descriptions-item>
						<el-descriptions-item label="参数来源">{{ overview.params?.source === 'uploaded' ? '文件夹上传' : '训练生成' }}</el-descriptions-item>
						<el-descriptions-item v-if="isAnomalyModel" label="基线文件">
							{{ overview.params?.baseline_file || '—' }}
						</el-descriptions-item>
						<el-descriptions-item v-if="isAnomalyModel" label="基线误报率">
							{{ overview.metrics?.baseline_false_positive_rate != null
								? (overview.metrics.baseline_false_positive_rate * 100).toFixed(2) + '%' : '—' }}
						</el-descriptions-item>
						<el-descriptions-item v-else label="类别标签">
							{{ (overview.labels || []).length ? `${overview.labels.length} 个` : '—' }}
						</el-descriptions-item>
					</el-descriptions>
					<div v-if="isAnomalyModel" class="hint mt">
						无监督判定：窗口 → 特征 → adtk 的 PCA 重构误差；阈值 = 正常窗口分数分布的
						{{ ((overview.params?.threshold_quantile ?? 0.995) * 100).toFixed(1) }}% 分位，
						推理时<strong>重构误差 &gt; 阈值即判「异常」</strong>，分数越大越异常。
					</div>
					<h4 class="mt">超参</h4>
					<el-descriptions :column="4" border size="small">
						<el-descriptions-item v-for="(v, k) in overview.params" :key="k" :label="String(k)">
							{{ Array.isArray(v) ? JSON.stringify(v) : (v === null || v === undefined ? '—' : v) }}
						</el-descriptions-item>
						<el-descriptions-item v-if="!Object.keys(overview.params || {}).length" label="超参">
							该模型没有超参记录（无监督模型，或从文件夹上传的产物）
						</el-descriptions-item>
					</el-descriptions>
				</template>
			</el-card>
		</el-tab-pane>

		<!-- ============ 训练 ============ -->
		<el-tab-pane label="训练" name="train">
			<el-alert type="warning" :closable="false" show-icon class="mb"
				title="训练是同步阻塞的：1DCNN 10 轮约 15 秒，cwt_cnn 50 轮约 25 秒，提交后请等待结果。" />
			<el-row :gutter="16">
				<el-col :xs="24" :md="12">
					<el-card shadow="never">
						<template #header><span>训练参数</span></template>
						<el-form label-width="130px" size="small">
							<el-form-item label="数据源">
								<el-select v-model="train.dataset_type" style="width: 100%">
									<el-option label="CWRU .mat（取 DE 通道）" value="matlab" />
									<el-option label="表格数据集（Excel/CSV，一文件一类别）" value="tabular" />
									<el-option label="自动（按目录内容判断）" value="auto" />
								</el-select>
							</el-form-item>
							<el-form-item label="数据集目录">
								<el-select v-model="train.dataset_dir" style="width: 100%" @change="syncSource">
									<el-option v-for="d in datasetOptions" :key="d.value" :label="d.label" :value="d.value" />
								</el-select>
							</el-form-item>
							<el-form-item label="信号列">
								<el-input v-model="train.signal_column" placeholder="表格数据留空=自动识别（如 振动幅值）" />
							</el-form-item>
							<el-form-item label="模型">
								<el-select v-model="train.model" style="width: 100%" @change="applyDefaults">
									<el-option label="算法模型1 · 1dcnn（TensorFlow/Keras）" value="1dcnn" />
									<el-option label="算法模型2 · cwt_cnn（PyTorch）" value="cwt_cnn" />
									<el-option label="算法模型3 · adtk（无监督，已搁置）" value="adtk" />
								</el-select>
							</el-form-item>
							<el-row :gutter="8">
								<el-col :span="8"><el-form-item label="轮次" label-width="50px"><el-input-number v-model="train.epochs" :min="1" :max="200" controls-position="right" style="width: 100%" /></el-form-item></el-col>
								<el-col :span="8"><el-form-item label="长度" label-width="50px"><el-input-number v-model="train.length" :min="64" :step="64" controls-position="right" style="width: 100%" /></el-form-item></el-col>
								<el-col :span="8"><el-form-item label="每类窗数" label-width="80px"><el-input-number v-model="train.number" :min="10" controls-position="right" style="width: 100%" /></el-form-item></el-col>
							</el-row>
							<el-row :gutter="8">
								<el-col :span="8"><el-form-item label="步长" label-width="50px"><el-input-number v-model="train.stride" :min="1" controls-position="right" style="width: 100%" /></el-form-item></el-col>
								<el-col :span="8"><el-form-item label="批大小" label-width="60px"><el-input-number v-model="train.batch_size" :min="1" controls-position="right" style="width: 100%" /></el-form-item></el-col>
								<el-col :span="8"><el-form-item label="种子" label-width="50px"><el-input-number v-model="train.seed" :min="0" controls-position="right" style="width: 100%" /></el-form-item></el-col>
							</el-row>
							<el-form-item label="数据集登记名">
								<el-input v-model="train.dataset" placeholder="写入 Datasets 表的名称" />
							</el-form-item>
							<el-form-item label="越界窗口">
								<el-switch v-model="train.strict" active-text="跳过（推荐，不补 NaN）" inactive-text="复刻旧脚本（补 NaN）" />
							</el-form-item>
							<el-button type="primary" :loading="training" @click="doTrain">
								{{ training ? `训练中… ${elapsed}s` : '开始训练' }}
							</el-button>
						</el-form>
					</el-card>
				</el-col>
				<el-col :xs="24" :md="12">
					<el-card shadow="never">
						<template #header><span>训练结果</span></template>
						<div v-if="!trainResult" class="empty">还没有训练结果</div>
						<template v-else>
							<el-descriptions :column="1" border size="small">
								<el-descriptions-item label="状态">
									<el-tag :type="trainResult.status === '成功' ? 'success' : 'danger'" size="small">{{ trainResult.status }}</el-tag>
									耗时 {{ trainResult.duration_sec }} 秒
								</el-descriptions-item>
								<el-descriptions-item label="产物">{{ trainResult.artifact?.weights }}（{{ trainResult.artifact?.version }}）</el-descriptions-item>
								<el-descriptions-item label="测试准确率">
									{{ fmt(trainResult.metrics?.test_accuracy) }} / loss {{ fmt(trainResult.metrics?.test_loss) }}
								</el-descriptions-item>
								<el-descriptions-item label="验证准确率">{{ fmt(trainResult.metrics?.val_accuracy) }}</el-descriptions-item>
								<el-descriptions-item label="训练/验证/测试">
									{{ trainResult.dataset_stats?.train_total }} / {{ trainResult.dataset_stats?.valid_total }} / {{ trainResult.dataset_stats?.test_total }}
								</el-descriptions-item>
								<el-descriptions-item label="跳过越界 / NaN">
									{{ trainResult.dataset_stats?.skipped_out_of_range_total ?? '—' }} / {{ trainResult.dataset_stats?.nan_windows_total ?? '—' }}
								</el-descriptions-item>
								<el-descriptions-item label="写库">
									<Tag :text="trainResult.db" />
								</el-descriptions-item>
							</el-descriptions>
							<div class="figs mt">
								<el-image v-for="f in trainResult.figures || []" :key="f.url" :src="fileUrl(f.url)"
									:preview-src-list="[fileUrl(f.url)]" fit="contain" class="fig" />
							</div>
							<el-collapse class="mt">
								<el-collapse-item title="classification_report">
									<pre class="pre">{{ trainResult.metrics?.classification_report || '—' }}</pre>
								</el-collapse-item>
							</el-collapse>
						</template>
					</el-card>
				</el-col>
			</el-row>
		</el-tab-pane>

		<!-- ============ 推理 ============ -->
		<el-tab-pane label="推理" name="predict">
			<el-row :gutter="16">
				<el-col :xs="24" :md="10">
					<el-card shadow="never">
						<template #header><span>推理参数</span></template>
						<el-form label-width="120px" size="small">
							<el-form-item label="模型">
								<el-select v-model="pred.model" style="width: 100%">
									<el-option label="算法模型1 · 1dcnn" value="1dcnn" />
									<el-option label="算法模型2 · cwt_cnn" value="cwt_cnn" />
									<el-option label="算法模型3 · adtk" value="adtk" />
								</el-select>
							</el-form-item>
							<el-form-item label="输入文件">
								<el-select v-model="pred.path" filterable style="width: 100%">
									<el-option v-for="f in fileOptions" :key="f.value" :label="f.label" :value="f.value" />
								</el-select>
							</el-form-item>
							<el-form-item label="信号列">
								<el-input v-model="pred.column" placeholder="表格文件留空=自动识别" />
							</el-form-item>
							<el-row :gutter="8">
								<el-col :span="8"><el-form-item label="起始窗" label-width="60px"><el-input-number v-model="pred.index" :min="0" controls-position="right" style="width: 100%" /></el-form-item></el-col>
								<el-col :span="8"><el-form-item label="窗口数" label-width="60px"><el-input-number v-model="pred.limit" :min="1" :max="500" controls-position="right" style="width: 100%" /></el-form-item></el-col>
								<el-col :span="8"><el-form-item label="top_k" label-width="60px"><el-input-number v-model="pred.top_k" :min="1" :max="10" controls-position="right" style="width: 100%" /></el-form-item></el-col>
							</el-row>
							<el-button type="primary" :loading="predicting" @click="doPredict">开始推理</el-button>
						</el-form>
					</el-card>
				</el-col>
				<el-col :xs="24" :md="14">
					<el-card shadow="never">
						<template #header><span>推理结果</span></template>
						<div v-if="!predResult" class="empty">还没有推理结果</div>
						<template v-else>
							<el-descriptions :column="1" border size="small">
								<el-descriptions-item label="模型 / 版本">{{ predResult.model }} · {{ predResult.version }} · {{ predResult.framework }}</el-descriptions-item>
								<el-descriptions-item label="样本数">{{ predResult.count }}（窗口长度 {{ predResult.input_len }}）</el-descriptions-item>
								<el-descriptions-item label="摘要">{{ JSON.stringify(predResult.summary) }}</el-descriptions-item>
								<el-descriptions-item label="写库"><Tag :text="predResult.db" /></el-descriptions-item>
							</el-descriptions>
							<el-table :data="predResult.predictions" size="small" class="mt">
								<el-table-column prop="index" label="窗口" width="70" />
								<el-table-column prop="predicted_label" label="预测" min-width="150" />
								<el-table-column label="置信度" width="110">
									<template #default="{ row }">{{ row.confidence != null ? Number(row.confidence).toFixed(4) : '—' }}</template>
								</el-table-column>
								<el-table-column label="异常分数" width="120">
									<template #default="{ row }">{{ fmtScore(row.anomaly_score) }}</template>
								</el-table-column>
								<el-table-column prop="actual_class" label="真实类别" width="100" />
								<el-table-column label="命中" width="90">
									<template #default="{ row }">
										<el-tag v-if="row.actual_class != null" :type="row.predicted_class === row.actual_class ? 'success' : 'danger'" size="small">
											{{ row.predicted_class === row.actual_class ? '命中' : '未命中' }}
										</el-tag>
										<span v-else>—</span>
									</template>
								</el-table-column>
							</el-table>
							<div class="figs mt">
								<el-image v-for="f in predResult.figures || []" :key="f.url" :src="fileUrl(f.url)"
									:preview-src-list="[fileUrl(f.url)]" fit="contain" class="fig" />
							</div>
						</template>
					</el-card>
				</el-col>
			</el-row>
		</el-tab-pane>

		<!-- ============ 训练记录 ============ -->
		<el-tab-pane label="训练记录" name="trainings">
			<el-card shadow="never">
				<template #header><span>最近训练（读库）</span><el-button link type="primary" style="float: right" @click="loadTrainings">刷新</el-button></template>
				<el-table :data="trainings" size="small" empty-text="还没有训练记录">
					<el-table-column prop="TrainingID" label="ID" width="70" />
					<el-table-column prop="ModelName" label="模型" width="110" />
					<el-table-column prop="DatasetName" label="数据集" width="160" />
					<el-table-column prop="TrainName" label="训练名" min-width="200" />
					<el-table-column prop="Epochs" label="轮次" width="80" />
					<el-table-column label="准确率" width="110"><template #default="{ row }">{{ fmt(row.Accuracy) }}</template></el-table-column>
					<el-table-column label="loss" width="110"><template #default="{ row }">{{ fmt(row.Loss) }}</template></el-table-column>
					<el-table-column label="状态" width="90">
						<template #default="{ row }"><el-tag :type="row.Status === '成功' ? 'success' : 'danger'" size="small">{{ row.Status }}</el-tag></template>
					</el-table-column>
					<el-table-column prop="StartedDate" label="开始时间" min-width="150" />
				</el-table>
			</el-card>
		</el-tab-pane>

		<!-- ============ 推理任务 ============ -->
		<el-tab-pane label="推理任务" name="tasks">
			<el-card shadow="never">
				<template #header><span>推理任务（读库）</span><el-button link type="primary" style="float: right" @click="loadTasks">刷新</el-button></template>
				<el-table :data="tasks" size="small" empty-text="还没有推理任务">
					<el-table-column prop="InferenceTaskID" label="ID" width="70" />
					<el-table-column prop="ModelName" label="模型" width="110" />
					<el-table-column prop="TaskType" label="类型" width="150" />
					<el-table-column prop="TrainingID" label="锚点训练" width="100" />
					<el-table-column prop="TargetDatasetID" label="目标数据集" width="110" />
					<el-table-column prop="Progress" label="进度" width="80" />
					<el-table-column label="状态" width="90">
						<template #default="{ row }"><el-tag :type="row.Status === '成功' ? 'success' : 'danger'" size="small">{{ row.Status }}</el-tag></template>
					</el-table-column>
					<el-table-column prop="CompletedDate" label="完成时间" min-width="150" />
					<el-table-column label="操作" width="90" fixed="right">
						<template #default="{ row }"><el-button link type="primary" @click="showTask(row.InferenceTaskID)">明细</el-button></template>
					</el-table-column>
				</el-table>
			</el-card>
		</el-tab-pane>
	</el-tabs>

	<!-- meta / 任务明细 弹窗 -->
	<!-- 登记弹窗：只用来「编辑」（新增登记改成从「上传模型」走，上传时自动登记） -->
	<el-dialog v-model="dialog.formVisible" :title="`编辑模型 ${dialog.originName || form.ModelName}`" width="520px">
		<el-form label-width="110px" size="small">
			<el-form-item label="模型名 *">
				<el-input v-model="form.ModelName" placeholder="唯一键，如 1DCNN" />
				<div class="hint">
					改名会一并重命名产物目录 <code>data/models/&lt;名&gt;/</code>，并更新库里已存的路径
				</div>
			</el-form-item>
			<el-form-item label="类型">
				<el-select v-model="form.ModelType" style="width: 100%">
					<el-option label="分类（Classification）" value="Classification" />
					<el-option label="异常检测（AnomalyDetection）" value="AnomalyDetection" />
					<el-option label="回归（Regression）" value="Regression" />
				</el-select>
			</el-form-item>
			<el-form-item label="接口"><el-input v-model="form.ApiEndpoint" placeholder="/predict" /></el-form-item>
			<el-form-item label="状态">
				<el-select v-model="form.Status" style="width: 100%" allow-create filterable>
					<el-option label="可运行" value="可运行" />
					<el-option label="未训练" value="未训练" />
					<el-option label="已停用" value="已停用" />
				</el-select>
			</el-form-item>
			<el-form-item label="说明"><el-input v-model="form.Description" type="textarea" :rows="3" /></el-form-item>
		</el-form>
		<div class="hint">编辑只改登记信息；换权重 / 加版本请用工具栏的「上传模型」。</div>
		<template #footer>
			<el-button @click="dialog.formVisible = false">取消</el-button>
			<el-button type="primary" @click="submitForm">保存</el-button>
		</template>
	</el-dialog>

	<!-- 上传弹窗：文件夹，或单个/多个模型文件 -->
	<el-dialog v-model="dialog.uploadVisible" title="上传模型（文件夹，或单个/多个文件）" width="560px">
		<el-form label-width="90px" size="small">
			<el-form-item label="模型名 *">
				<el-input v-model="uploadForm.name" placeholder="唯一键，如 1DCNN；同名会新增一个版本" />
			</el-form-item>
			<el-form-item label="说明"><el-input v-model="uploadForm.description" placeholder="可选" /></el-form-item>
		</el-form>
		<div class="hint">
			权重文件必须有：<code>.h5 .keras .pt .pth .pkl</code>；<code>scaler.npz</code>、<code>meta.json</code> 可选。
			服务端会先<strong>判断这是不是一个模型</strong>（看文件内容，不只看后缀），再自动读出输入长度与类别数。
		</div>
		<div class="mt" style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap">
			<input ref="folderInput" type="file" webkitdirectory directory multiple style="display: none" @change="onPick" />
			<input ref="fileInput" type="file" multiple accept=".h5,.keras,.pt,.pth,.pkl,.pickle,.npz,.json" style="display: none" @change="onPick" />
			<el-button size="small" @click="pickFolder">选择整个文件夹</el-button>
			<el-button size="small" @click="pickFiles">选择模型文件（可多选）</el-button>
			<el-button v-if="picked.length" size="small" text @click="clearPicked">清空</el-button>
		</div>
		<el-table v-if="picked.length" :data="picked" size="small" max-height="150" class="mt">
			<el-table-column prop="name" label="文件" min-width="200" />
			<el-table-column prop="size_kb" label="KB" width="90" />
			<el-table-column label="识别为" width="100">
				<template #default="{ row }">
					<el-tag :type="row.kind === '权重' ? 'success' : row.kind === '其他' ? 'info' : 'warning'" size="small">{{ row.kind }}</el-tag>
				</template>
			</el-table-column>
		</el-table>
		<div v-if="uploadMsg" class="hint mt" style="white-space: pre-line">{{ uploadMsg }}</div>
		<template #footer>
			<el-button @click="dialog.uploadVisible = false">取消</el-button>
			<el-button type="primary" :loading="uploading" @click="doUploadModel">上传</el-button>
		</template>
	</el-dialog>

	<el-dialog v-model="dialog.meta" :title="`${dialog.model} 的 meta.json`" width="70%">
		<pre class="pre">{{ dialog.metaText }}</pre>
	</el-dialog>
	<el-dialog v-model="dialog.task" :title="`推理任务 #${dialog.taskId}`" width="80%">
		<el-descriptions v-if="dialog.taskData" :column="1" border size="small">
			<el-descriptions-item label="任务名">{{ dialog.taskData.TaskName }}</el-descriptions-item>
			<el-descriptions-item label="锚点 / 目标">TrainingID={{ dialog.taskData.TrainingID }}　DatasetID={{ dialog.taskData.TargetDatasetID }}</el-descriptions-item>
			<el-descriptions-item label="输入 / 输出">{{ dialog.taskData.InputPath }} → {{ dialog.taskData.OutputPath }}</el-descriptions-item>
			<el-descriptions-item label="结果摘要">{{ dialog.taskData.ResultSummary }}</el-descriptions-item>
		</el-descriptions>
		<el-table :data="dialog.taskData?.results || []" size="small" class="mt" max-height="380">
			<el-table-column prop="ResultID" label="结果ID" width="80" />
			<el-table-column prop="RowIdentifier" label="样本" min-width="200" />
			<el-table-column prop="SampleIndex" label="窗口" width="70" />
			<el-table-column prop="PredictedLabel" label="预测" min-width="140" />
			<el-table-column label="置信度" width="100"><template #default="{ row }">{{ fmt(row.Confidence) }}</template></el-table-column>
			<el-table-column prop="ActualClass" label="真实" width="80" />
		</el-table>
	</el-dialog>
	</div>
</template>

<script setup lang="ts" name="platformModel">
import { computed, defineComponent, h, onMounted, reactive, ref } from 'vue';
import { useRoute } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { platformApi, fileUrl } from '/@/api/platform';

/** 写库回执的小标签 */
const Tag = defineComponent({
	props: { text: { type: Object, default: () => ({}) } },
	setup(props) {
		return () => {
			const d: any = props.text || {};
			return d.written
				? h('span', [`已写入（${d.dialect}）ID=${d.TrainingID ?? d.InferenceTaskID ?? '—'}`])
				: h('span', { style: 'color:#f56c6c' }, d.error || '未写入');
		};
	},
});

const route = useRoute();
const tab = ref<string>((route.query.tab as string) || 'list');
const fmt = (v: any) => (v === null || v === undefined ? '—' : Number(v).toFixed(4));
/** Models.ModelType 库里存英文，界面统一显示中文；不认识的值原样显示，别把信息吞掉 */
const TYPE_LABELS: Record<string, string> = { Classification: '分类', AnomalyDetection: '异常检测', Regression: '回归' };
const typeLabel = (v?: string) => {
	const raw = String(v || '');
	const hit = Object.keys(TYPE_LABELS).find((k) => k.toLowerCase() === raw.toLowerCase());
	return hit ? TYPE_LABELS[hit] : (raw || '—');
};

const artifacts = ref<any[]>([]);
const dbModels = ref<any[]>([]);
const trainings = ref<any[]>([]);
const tasks = ref<any[]>([]);
const datasets = ref<Record<string, any>>({});
const overview = ref<any>(null);
const form = reactive<any>({ ModelName: '', Description: '', ModelType: 'Classification', ApiEndpoint: '/predict', Status: '' });
const dialog = reactive<any>({ meta: false, model: '', metaText: '', task: false, taskId: 0, taskData: null,
	formVisible: false, originName: '', uploadVisible: false });
/** 上传弹窗自己的字段，跟「新增/编辑」登记弹窗解耦，互不影响 */
const uploadForm = reactive<any>({ name: '', description: '' });

const datasetOptions = computed(() =>
	Object.entries(datasets.value).map(([k, d]: any) => ({
		label: `${k}　[${d.dataset_type === 'tabular' ? '表格' : 'mat'}，${d.file_count ?? 0} 个文件]`,
		value: d.dataset_dir,
		kind: d.dataset_type,
	}))
);
const fileOptions = computed(() => {
	const out: any[] = [];
	Object.entries(datasets.value).forEach(([k, d]: any) => {
		(d.files || []).filter((f: any) => f.on_disk).forEach((f: any) => {
			out.push({ label: `${k} / ${f.filename}（${f.label}）`, value: `${d.dataset_dir}\\${f.filename}` });
		});
	});
	return out;
});

const train = reactive<any>({ model: '1dcnn', dataset_type: 'matlab', dataset_dir: '', signal_column: '',
	epochs: 10, length: 784, number: 600, stride: 150, batch_size: 128, seed: 42, dataset: 'CWRU-0HP', strict: true });
const pred = reactive<any>({ model: '1dcnn', path: '', column: '', index: 0, limit: 4, top_k: 3 });

const training = ref(false);
const predicting = ref(false);
const elapsed = ref(0);
const trainResult = ref<any>(null);
const predResult = ref<any>(null);

/** 模型清单 = Models 表登记信息 与 产物参数 合并成一行（1dcnn ↔ 1DCNN 用忽略大小写匹配） */
const mergedRows = computed(() => {
	const map = new Map<string, any>();
	(dbModels.value || []).forEach((r: any) => {
		map.set(String(r.ModelName).toLowerCase(), { name: r.ModelName, reg: r, art: null });
	});
	(artifacts.value || []).forEach((a: any) => {
		const key = String(a.model).toLowerCase();
		if (!map.has(key)) map.set(key, { name: a.model, reg: null, art: null });
		map.get(key).art = a;              // artifacts 按版本升序返回，覆盖后即最新版本
	});
	return [...map.values()];
});
/** 最新一版产物 + 是不是异常检测类模型（决定详情卡展示哪一套字段） */
const latestArt = computed(() => {
	const versions = overview.value?.artifact_versions || [];
	return versions[versions.length - 1] || {};
});
const isAnomalyModel = computed(() => {
	const reg = overview.value?.registration || {};
	return String(reg.ModelType || '').toLowerCase() === 'anomalydetection'
		|| latestArt.value.framework === 'adtk'
		|| (overview.value?.params || {}).detector != null;
});
/** 阈值/分数是 1e-3 量级的小数，用科学计数法更好读 */
const fmtThreshold = (v: any) => (v === null || v === undefined ? '—' : Number(v).toExponential(3));
/** 异常分数：大数用定点、小数用科学计数法，别把 4.5e-5 显示成 0.0000 */
const fmtScore = (v: any) => {
	if (v === null || v === undefined) return '—';
	const n = Number(v);
	return Math.abs(n) >= 0.001 ? n.toFixed(4) : n.toExponential(3);
};
/** 框架名压缩版（列表格子窄，tensorflow-keras 太长撑不下） */
const FW_LABELS: Record<string, string> = { 'tensorflow-keras': 'Keras', pytorch: 'PyTorch', adtk: 'ADTK' };
const fwLabel = (v?: string) => FW_LABELS[String(v || '')] || (v || '—');
/** 列表行是不是异常检测（列表数据来自 /models，没有 overview，只能靠 registration + 产物判断） */
const isAnomalyRow = (row: any) => String(row?.reg?.ModelType || '').toLowerCase() === 'anomalydetection'
	|| row?.art?.framework === 'adtk'
	|| (row?.art?.params || {}).detector != null;
/** 列表「指标」列：分类/回归看测试准确率，异常检测看基线误报率（无监督没有准确率） */
const rowMetric = (row: any): { value: string; label: string } => {
	const metrics = row?.art?.metrics || {};
	if (isAnomalyRow(row)) {
		const rate = metrics.baseline_false_positive_rate;
		return { value: rate === null || rate === undefined ? '—' : `${(Number(rate) * 100).toFixed(2)}%`,
			label: '基线误报率' };
	}
	const accuracy = metrics.test_accuracy;
	return { value: accuracy === null || accuracy === undefined ? '—' : fmt(accuracy), label: '测试准确率' };
};
const labelRows = computed(() =>
	((overview.value?.labels as string[]) || []).map((label, id) => ({ id, label }))
);

const loadModels = async () => {
	const res: any = await platformApi.models();
	artifacts.value = res.artifacts || [];
	dbModels.value = res.db_models || [];
};
const onRowClick = (row: any) => { if (row?.name) loadOverview(row.name); };
const loadOverview = async (name: string) => {
	overview.value = (await platformApi.modelOverview(name)) as any;
};

/** 编辑登记信息（新增登记已取消：上传模型时后端会自动登记一行） */
const openEdit = (row: any) => {
	const r = row.reg || {};
	dialog.originName = row.name;                 // 改名要拿老名字当 URL，所以单独记一份
	Object.assign(form, { ModelName: row.name, Description: r.Description || '', ModelType: r.ModelType || 'Classification',
		ApiEndpoint: r.ApiEndpoint || '', Status: r.Status || '' });
	dialog.formVisible = true;
};
const submitForm = async () => {
	if (!form.ModelName) { ElMessage.warning('模型名不能为空'); return; }
	try {
		const payload: any = { Description: form.Description, ModelType: form.ModelType,
			ApiEndpoint: form.ApiEndpoint, Status: form.Status };
		if (form.ModelName !== dialog.originName) payload.NewModelName = form.ModelName;   // 改名：连产物目录一起搬
		const res: any = await platformApi.updateModel(dialog.originName, payload);
		const rn = res?.rename;
		ElMessage.success(rn
			? `已改名为 ${rn.to}` + (rn.artifact_dir_moved ? `（产物目录已搬到 ${rn.artifact_dir}）` : '（无产物目录，只改了登记）')
				+ (rn.note ? `；${rn.note}` : '')
			: `已更新模型 ${form.ModelName}`);
	} catch (e: any) {
		ElMessage.error('保存失败：' + (e?.response?.data?.error || e?.message || e));
		return;                                   // 失败就别关弹窗，让用户改完再提交
	}
	dialog.formVisible = false;
	await loadModels();
	if (overview.value) await loadOverview(overview.value.model);
};

/** 删除：先查引用，把"被哪些表引用多少行"放进确认框 */
const removeModel = async (row: any) => {
	let refs: any = {};
	try { refs = (await platformApi.modelReferences(row.name)) as any; } catch (e) { /* 未登记时忽略 */ }
	const detail = refs?.references ? JSON.stringify(refs.references) : '（无引用信息）';
	const hasRefs = (refs?.total || 0) > 0;
	try {
		await ElMessageBox.confirm(
			`将删除 Models 表里 ${row.name} 的登记行。\n引用情况：${detail}` +
			(hasRefs ? '\n⚠ 仍被引用：确认后将连带删除这些训练/推理记录（不可撤销）' : ''),
			hasRefs ? '危险操作（有引用）' : '确认删除', { type: 'warning' });
	} catch { return; }
	const res: any = await platformApi.deleteModelRecord(row.name, hasRefs);
	ElMessage.success(`已删除 ${res.deleted}${res.cascaded ? '（已连带清理引用记录）' : ''}`);
	if (overview.value?.model === row.name) overview.value = null;
	await loadModels();
};

/** 删除某个产物版本（只删文件，不动库表登记） */
const removeVersionFile = async (name: string, version: string) => {
	await ElMessageBox.confirm(`删除 ${name} 的产物版本 ${version}？权重/scaler/meta 一并删除。`, '危险操作', { type: 'warning' });
	const res: any = await platformApi.deleteVersion(name, version);
	ElMessage.success(`已删除 ${res.deleted}（${res.files} 个文件）`);
	await Promise.all([loadModels(), loadOverview(name)]);
};

/** 上传模型：文件夹或单个/多个文件都行；不再手填输入长度/类别数——服务端探测后自动识别 */
const folderInput = ref<HTMLInputElement>();
const fileInput = ref<HTMLInputElement>();
const picked = ref<{ name: string; size_kb: number; kind: string }[]>([]);
const uploading = ref(false);
const uploadMsg = ref('');
const pickKind = (n: string) =>
	/\.(h5|keras|pt|pth|pkl|pickle)$/i.test(n) ? '权重'
		: n === 'scaler.npz' ? 'scaler' : n === 'meta.json' ? 'meta' : '其他';
/** 两个隐藏的原生 input（浏览器没有"文件夹 + 文件"合一的入口），由两个按钮分别去点 */
const pickFolder = () => folderInput.value?.click();
const pickFiles = () => fileInput.value?.click();
/** 两个选择器里选中的文件合起来 */
const pickedFiles = (): File[] => [
	...Array.from(folderInput.value?.files || []),
	...Array.from(fileInput.value?.files || []),
];
const onPick = () => {
	picked.value = pickedFiles().map((f) => ({
		name: f.name, size_kb: Number((f.size / 1024).toFixed(1)), kind: pickKind(f.name),
	}));
	uploadMsg.value = picked.value.some((f) => f.kind === '权重')
		? '' : '⚠ 没识别到权重文件（.h5/.keras/.pt/.pth/.pkl），服务端会判定「不是一个模型」并拒绝';
};
const clearPicked = () => {
	if (folderInput.value) folderInput.value.value = '';
	if (fileInput.value) fileInput.value.value = '';
	picked.value = [];
	uploadMsg.value = '';
};
/** 工具栏「上传模型」：每次打开都是干净的空表单 */
const openUpload = () => {
	clearPicked();
	uploadForm.name = '';
	uploadForm.description = '';
	dialog.uploadVisible = true;
};
const doUploadModel = async () => {
	const files = pickedFiles();
	if (!files.length) { ElMessage.warning('先点「选择整个文件夹」或「选择模型文件」'); return; }
	if (!uploadForm.name) { ElMessage.warning('先填模型名'); return; }
	const fd = new FormData();
	fd.append('name', uploadForm.name);
	fd.append('description', uploadForm.description || '');
	files.forEach((f) => fd.append('file', f, f.name));
	uploading.value = true;
	uploadMsg.value = '';
	try {
		const res: any = await platformApi.uploadModel(fd);
		uploadMsg.value = [
			`已上传 → ${res.directory}`,
			`框架 ${res.framework}`,
			`权重 ${res.probe?.weights || res.weights}`,
			`input_len=${res.input_len ?? '未识别(推理按默认 784)'}`,
			`${res.num_classes ?? '?'} 类`,
			res.meta_generated ? '已自动生成 meta.json' : '使用文件夹里的 meta.json',
		].join(' · ') + (res.warnings?.length ? `\n⚠ ${res.warnings.join('；')}` : '');
		ElMessage.success('模型上传成功');
		clearPicked();
		await loadModels();
		await loadOverview(res.model);
		dialog.uploadVisible = false;
	} catch (e: any) {
		const data = e?.response?.data;
		const detail = (data?.detail || []).map((d: any) => `${d.filename}：${d.reason}`).join('\n');
		uploadMsg.value = '上传失败：' + (data?.error || e?.message || e) + (detail ? `\n${detail}` : '');
		ElMessage.error('上传失败：' + (data?.error || e?.message || e));
	} finally {
		uploading.value = false;
	}
};

const loadTrainings = async () => { trainings.value = ((await platformApi.trainings(20)) as any).trainings || []; };
const loadTasks = async () => { tasks.value = ((await platformApi.tasks(20)) as any).tasks || []; };
const loadDatasets = async () => {
	datasets.value = (await platformApi.datasets()) as any;
	if (!train.dataset_dir) {
		const first = datasetOptions.value[0];
		if (first) { train.dataset_dir = first.value; train.dataset_type = (first as any).kind; }
	}
};

const syncSource = () => {
	const hit = datasetOptions.value.find((o) => o.value === train.dataset_dir);
	if (hit) train.dataset_type = (hit as any).kind;
};
const applyDefaults = () => {
	if (train.model === '1dcnn') { train.epochs = 10; train.number = 600; }
	else if (train.model === 'cwt_cnn') { train.epochs = 50; train.number = 300; }
	else { train.epochs = 1; }
};

const showMeta = async (name: string) => {
	const res: any = await platformApi.artifact(name);
	dialog.model = name;
	dialog.metaText = JSON.stringify(res, null, 2);
	dialog.meta = true;
};
const showTask = async (id: number) => {
	dialog.taskId = id;
	dialog.taskData = await platformApi.taskDetail(id);
	dialog.task = true;
};

const doTrain = async () => {
	training.value = true;
	elapsed.value = 0;
	const timer = setInterval(() => (elapsed.value += 1), 1000);
	try {
		const payload: any = { ...train };
		payload.strict = !!train.strict;
		if (!payload.signal_column) delete payload.signal_column;
		trainResult.value = await platformApi.train(payload);
		await Promise.all([loadModels(), loadTrainings()]);
	} finally {
		clearInterval(timer);
		training.value = false;
	}
};

const doPredict = async () => {
	predicting.value = true;
	try {
		const payload: any = { ...pred };
		if (!payload.column) delete payload.column;
		predResult.value = await platformApi.predict(payload);
		await loadTasks();
	} finally {
		predicting.value = false;
	}
};

onMounted(async () => {
	await Promise.all([loadModels(), loadTrainings(), loadTasks(), loadDatasets()]);
});
</script>

<style scoped lang="scss">
.mb { margin-bottom: 16px; }
.mt { margin-top: 16px; }
.empty { color: var(--el-text-color-secondary); text-align: center; padding: 24px 0; }
.pre { max-height: 320px; overflow: auto; background: #141413; color: #ede9e0; padding: 12px; border-radius: 6px; font-size: 12px; }
.figs { display: flex; flex-wrap: wrap; gap: 12px; }
.fig { width: 260px; height: 170px; border: 1px solid var(--el-border-color); border-radius: 6px; background: #fff; }
</style>
