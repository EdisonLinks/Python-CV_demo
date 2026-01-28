// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('main'));

var option = {
    title: {
        text: '【门禁】入口访客流量趋势'
    },
    tooltip: {
        trigger: 'axis'
    },
    xAxis: {
        type: 'category',
        data: [] // 初始为空
    },
    yAxis: {
        type: 'value',
        min: 0, // 设置Y轴最小值
        max: 100 // 设置Y轴最大值
    },
    series: [{
        data: [],
        type: 'line',
        animationDurationUpdate: 1000 // 动画持续时间
    }]
};

// 使用刚指定的配置项和数据显示图表。
myChart.setOption(option);


var data = []; // 初始数据集
var xAxisData = []; // x轴数据集

for (var i = 0; i < 20; i++) { // 生成一些初始数据和x轴标签
    data.push(0);
    xAxisData.push('时刻' + i);
}

// WebSocket连接设置
var ws = new WebSocket('ws://localhost:8765'); // 替换为你的WebSocket服务器地址
ws.onmessage = function(event) {
    var newData = JSON.parse(event.data); // 假设服务器发送的数据是JSON格式的数组形式 [xData, yData]
    var xData = newData[0]; // 获取X轴数据（时间戳等）
    var yData = newData[1]; // 获取Y轴数据（数值）

    data.push(yData);
    xAxisData.push(xData);
    data.shift(); // 移除数组开始的数据点
    xAxisData.shift(); // 移除x轴开始的标签

    myChart.setOption({
        xAxis: {
            data: xAxisData
        },
        series: [{
            data: data
        }]
    });
};

