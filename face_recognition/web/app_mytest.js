// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('main'));


// data= [36,161,54,42,88];  //验证数据1
// data2= [79,184,156,48,37];  //验证数据2

var data = []; // 初始数据集,用来保存Y值
var data2 = [];// 用来保存Y的值
var xAxisData = []; // x轴数据集，双线是一样的


//  初始填充一些数据
for (var i = 0; i < 20; i++) { // 生成一些初始数据和x轴标签
    data.push(0);   //初始化数据设为0，
    data2.push(0)
    xAxisData.push('时刻' + i); //生成初始的x轴标签
}


var option = {
  title: {
    text: '当前签到人数流量趋势'
  },
  tooltip: {
    trigger: 'axis'         //鼠标悬浮提示框
  },
  toolbox: {  //工具箱：保存图片的
    feature: {
      saveAsImage: {}
    }
  },
  legend: {
    data: ['老师', '学生']    // 图例显示，必须要与series中的字典相对应
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,  //坐标轴两边不留白
    data: []   //x轴数据，由程序动态填充
  },
  yAxis: {
    type: 'value',
    min: 0, // 设置Y轴最小值
    max: 150 // 设置Y轴最大值

  },
  series: [
    {
      name: '老师',
      type: 'line',
      // stack: 'Total',  // 独立显示应该删除堆叠设置Total
      // data: [],                           //此处表示‘目标人物’出现的数据统计
      data: data,
      animationDurationUpdate:1000  // 持续时间
    },
    {
      name: '学生',
      type: 'line',
      // stack: 'Total',
      data: data2,                         //此处表示‘路人’出现的数据统计
      animationDurationUpdate:1000  // 持续时间

    }
  ]
};


// 使用刚指定的配置项和数据显示图表。
myChart.setOption(option);



// WebSocket连接设置
var ws = new WebSocket('ws://localhost:8765'); // 替换为你的WebSocket服务器地址，用于接收实时数据
ws.onmessage = function(event) {
    var newData = JSON.parse(event.data);
    var xData = newData[0]; // 获取X轴数据（时间戳等）
    var yData = newData[1]; // 获取Y轴数据（访客总数）

    // 由于只接收2个元素，将访客总数同时作为两组数据显示
    var y1Data = yData; // 目标人物数量
    var y2Data = yData; // 路人数量
   //滑动窗口
    data.push(y1Data);  // 新数据添加到数组中
    data2.push(y2Data);
    xAxisData.push(xData);

    data.shift(); // 移除数组开始的数据点
    data2.shift();
    xAxisData.shift(); // 移除x轴开始的标签

    myChart.setOption({
        xAxis: {
            data: xAxisData
        },
        series: [
            {name:'老师', data: data},   //线条1
            {name:'学生', data: data2}   //线条2
        ]
    });
};





