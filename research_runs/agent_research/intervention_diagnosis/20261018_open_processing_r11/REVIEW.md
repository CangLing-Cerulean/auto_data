# 聚焦只读审查及定向回归

审查者为check子代理；主代理负责修复与最终验收。下列记录为审查者返回的事实与边界，不把mock当真实服务器执行。

## 数值程序与隔离

初始审查用纯人工数据复现np.sqrt(a,where=False)读到曾写入1234567的释放块，五次计数7964/9998/9998/9998/9998。禁止所有where关键字及既知未初始化分配入口后，sqrt、np.clip、数组clip、pad路径全部拒绝；位置np.where正常。

初始状态接口也复现longdouble/clongdouble scalar通过宽度检查后仍不能JSON编码，以及10**5000触发JSON数字位数限制。signed64边界拒绝通过；首次宽度修复仍未解决Windows同宽扩展别名，第二次改显式Python数值转换，program的15项检查含该往返通过。worker序列化异常已由独立mock验证返回INVALID_PROGRAM。

未发现其他具体可复现反射/全局状态逃逸。该表述不等于完整安全证明；Linux seccomp另有真实合成检查收据，未接触真实数组。

## 运行接入与编排

初轮审查四文件：open_processing_runtime、open_processing_trial、preflight_open_processing、manage_open_processing。纯Python mock模拟16尝试/32调用，通过交替次序、本组历史、普通参考更新、私有缓存字段过滤及SEPARATE专属账本。静态未发现新增未来真值泄露、窗口错配或A拟合/B选点/C评分错误；缓存和数值失败整次拒绝未见确定缺陷。

两个已复现缺陷：直接runtime.query入口无预检收据仍可运行；调用含error事件但exit0+turn.completed仍可PASS。已在正式冻结前修复。

定向回归（纯Python mock、退出码0）：

- runtime有效绑定通过；缺失/FAIL收据、错误lock hash、preflight hash、父路径/资产/数量共11条错误路径均拒绝。
- trial有效绑定通过；9条错误锁/收据路径均拒绝。
- item error、顶层error、turn.failed均保留并判FAILED；干净完成事件PASS。
- 实际preflight收据构造代码向两份收据写入当前lock hash，且reuse绑定正确preflight hash。

审查未联网、未访问真实数组、未调用LLM或训练，也未重复已有GPU整套测试。真实worker IPC和恒等父复用仍由部署后预检验收。独立开发审计入口另由主代理审阅并在执行前冻结。
