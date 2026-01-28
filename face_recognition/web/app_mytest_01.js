// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('main'));
var pieChart = echarts.init(document.getElementById('pie-chart'));

// 初始数据集
var data = []; // 当前签到人数
var xAxisData = [];
// 添加时间格式化函数
function formatTime(date) {
    var hours = date.getHours().toString().padStart(2, '0');
    var minutes = date.getMinutes().toString().padStart(2, '0');
    return hours + ':' + minutes;
}
// 初始填充一些数据
for (var i = 0; i < 20; i++) {
    data.push(0);
    xAxisData.push('时刻' + i);
}

// 折线图配置
var option = {
  title: {
    text: '当前签到人数流量趋势'
  },
  tooltip: {
    trigger: 'axis'
  },
  toolbox: {
    feature: {
      saveAsImage: {}
    }
  },
  legend: {
    data: ['总签到人数']
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: xAxisData
  },
  yAxis: {
    type: 'value',
    min: 0,
    max: 10
  },
  series: [
    {
      name: '总签到人数',
      type: 'line',
      data: data,
      animationDurationUpdate: 1000
    }
  ]
};

// 饼图配置
var pieOption = {
    title: {
        text: '班级出勤率',
        left: 'center'
    },
    tooltip: {
        trigger: 'item'
    },
    legend: {
        orient: 'vertical',
        left: 'left'
    },
    series: [{
        name: '出勤情况',
        type: 'pie',
        radius: '50%',
        data: [
            {value: 0, name: '老师'},
            {value: 0, name: '学生'}
        ],
        emphasis: {
            itemStyle: {
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
        }
    }]
};

// 使用配置项和数据显示图表
myChart.setOption(option);
pieChart.setOption(pieOption);

// WebSocket连接设置
var ws = new WebSocket('ws://localhost:8765');

ws.onopen = function(event) {
    console.log("WebSocket连接已建立");
    document.getElementById('recognized-list').innerHTML = '<h3>已签到人员</h3><p>正在连接服务器...</p>';
};

ws.onmessage = function(event) {
    try {
        var newData = JSON.parse(event.data);
        console.log("收到数据:", newData);

        // 根据后端发送的实际数据格式处理数据
        var currentTime = new Date(); // 获取当前时间
        var xData = formatTime(currentTime); // 格式化为 HH:mm
        
        // 处理不同格式的数据：后端可能发送两种格式
        var yData;
        if (Array.isArray(newData)) {
            // 如果是数组格式 [时间, 总人数]
            yData = newData[1] || 0;
        } else {
            // 如果是对象格式，从后端check_test01.py中获取数据
            yData = (newData.total_visitors !== undefined) ? newData.total_visitors : 0;
        }

        data.push(yData);
        xAxisData.push(xData);

        data.shift();
        xAxisData.shift();

        myChart.setOption({
            xAxis: {
                data: xAxisData
            },
            series: [
                {data: data}
            ]
        });

        // 更新饼图数据 - 从对象格式获取老师和学生数量
        var teacherCount = 0;
        var studentCount = 0;
        if (!Array.isArray(newData) && newData.teachers !== undefined && newData.students !== undefined) {
            teacherCount = newData.teachers || 0;
            studentCount = newData.students || 0;
        }

        pieChart.setOption({
            series: [{
                data: [
                    {value: teacherCount, name: '老师'},
                    {value: studentCount, name: '学生'}
                ]
            }]
        });

        // 更新已识别人员列表 - 从对象格式获取已识别列表
        var recognizedList = [];
        if (!Array.isArray(newData) && newData.recognized_list) {
            recognizedList = newData.recognized_list || [];
        }
        updateRecognizedList(recognizedList);
    } catch (e) {
        console.error("数据处理错误:", e);
        console.log("原始数据:", event.data);
    }
};

ws.onerror = function(error) {
    console.error("WebSocket错误:", error);
    document.getElementById('recognized-list').innerHTML = '<h3>连接错误</h3><p>无法连接到服务器，请确保后端程序正在运行</p>';
};

ws.onclose = function(event) {
    console.log("WebSocket连接已关闭");
    document.getElementById('recognized-list').innerHTML = '<h3>连接断开</h3><p>与服务器的连接已断开</p>';
};

function updateRecognizedList(recognizedList) {
    var listHtml = '<h3>已签到人员</h3><ul>';
    if (recognizedList && recognizedList.length > 0) {
        recognizedList.forEach(function(name) {
            listHtml += '<li>' + name + '</li>';
        });
    } else {
        listHtml += '<li>暂无签到人员</li>';
    }
    listHtml += '</ul>';
    document.getElementById('recognized-list').innerHTML = listHtml;
}

// 添加文字转语音功能
function speakMessage(message) {
    if ('speechSynthesis' in window) {
        var utterance = new SpeechSynthesisUtterance(message);
        utterance.lang = 'zh-CN';
        speechSynthesis.speak(utterance);
    }
}
