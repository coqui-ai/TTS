# Authors:
#   2019.5 Zhiyang Zhou (https://github.com/Joee1995/chn_text_norm.git)
#   2019.9 - 2022 Jiayu DU

import argparse
import csv
import os
import re
import string
import sys

# fmt: off

# ================================================================================ #
#                                    basic constant
# ================================================================================ #
CHINESE_DIGIS = "零一二三四五六七八九"
BIG_CHINESE_DIGIS_SIMPLIFIED = "零壹贰叁肆伍陆柒捌玖"
BIG_CHINESE_DIGIS_TRADITIONAL = "零壹貳參肆伍陸柒捌玖"
SMALLER_BIG_CHINESE_UNITS_SIMPLIFIED = "十百千万"
SMALLER_BIG_CHINESE_UNITS_TRADITIONAL = "拾佰仟萬"
LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED = "亿兆京垓秭穰沟涧正载"
LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL = "億兆京垓秭穰溝澗正載"
SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED = "十百千万"
SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL = "拾佰仟萬"

ZERO_ALT = "〇"
ONE_ALT = "幺"
TWO_ALTS = ["两", "兩"]

POSITIVE = ["正", "正"]
NEGATIVE = ["负", "負"]
POINT = ["点", "點"]
# PLUS = [u'加', u'加']
# SIL = [u'杠', u'槓']

FILLER_CHARS = ["呃", "啊"]

ER_WHITELIST = (
    "(儿女|儿子|儿孙|女儿|儿媳|妻儿|"
    "胎儿|婴儿|新生儿|婴幼儿|幼儿|少儿|小儿|儿歌|儿童|儿科|托儿所|孤儿|"
    "儿戏|儿化|台儿庄|鹿儿岛|正儿八经|吊儿郎当|生儿育女|托儿带女|养儿防老|痴儿呆女|"
    "佳儿佳妇|儿怜兽扰|儿无常父|儿不嫌母丑|儿行千里母担忧|儿大不由爷|苏乞儿)"
)
ER_WHITELIST_PATTERN = re.compile(ER_WHITELIST)

# 中文数字系统类型
NUMBERING_TYPES = ["low", "mid", "high"]

CURRENCY_NAMES = "(人民币|美元|日元|英镑|欧元|马克|法郎|加拿大元|澳元|港币|先令|芬兰马克|爱尔兰镑|" "里拉|荷兰盾|埃斯库多|比塞塔|印尼盾|林吉特|新西兰元|比索|卢布|新加坡元|韩元|泰铢)"
CURRENCY_UNITS = "((亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|)元|(亿|千万|百万|万|千|百|)块|角|毛|分)"
COM_QUANTIFIERS = (
    "(匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|"
    "砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|"
    "针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|"
    "毫|厘|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|"
    "盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|旬|"
    "纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块)"
)


# Punctuation information are based on Zhon project (https://github.com/tsroten/zhon.git)
CN_PUNCS_STOP = "！？｡。"
CN_PUNCS_NONSTOP = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏·〈〉-"
CN_PUNCS = CN_PUNCS_STOP + CN_PUNCS_NONSTOP

PUNCS = CN_PUNCS + string.punctuation
PUNCS_TRANSFORM = str.maketrans(PUNCS, " " * len(PUNCS), "")  # replace puncs with space


# https://zh.wikipedia.org/wiki/全行和半行
QJ2BJ = {
    "　": " ",
    "！": "!",
    "＂": '"',
    "＃": "#",
    "＄": "$",
    "％": "%",
    "＆": "&",
    "＇": "'",
    "（": "(",
    "）": ")",
    "＊": "*",
    "＋": "+",
    "，": ",",
    "－": "-",
    "．": ".",
    "／": "/",
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
    "：": ":",
    "；": ";",
    "＜": "<",
    "＝": "=",
    "＞": ">",
    "？": "?",
    "＠": "@",
    "Ａ": "A",
    "Ｂ": "B",
    "Ｃ": "C",
    "Ｄ": "D",
    "Ｅ": "E",
    "Ｆ": "F",
    "Ｇ": "G",
    "Ｈ": "H",
    "Ｉ": "I",
    "Ｊ": "J",
    "Ｋ": "K",
    "Ｌ": "L",
    "Ｍ": "M",
    "Ｎ": "N",
    "Ｏ": "O",
    "Ｐ": "P",
    "Ｑ": "Q",
    "Ｒ": "R",
    "Ｓ": "S",
    "Ｔ": "T",
    "Ｕ": "U",
    "Ｖ": "V",
    "Ｗ": "W",
    "Ｘ": "X",
    "Ｙ": "Y",
    "Ｚ": "Z",
    "［": "[",
    "＼": "\\",
    "］": "]",
    "＾": "^",
    "＿": "_",
    "｀": "`",
    "ａ": "a",
    "ｂ": "b",
    "ｃ": "c",
    "ｄ": "d",
    "ｅ": "e",
    "ｆ": "f",
    "ｇ": "g",
    "ｈ": "h",
    "ｉ": "i",
    "ｊ": "j",
    "ｋ": "k",
    "ｌ": "l",
    "ｍ": "m",
    "ｎ": "n",
    "ｏ": "o",
    "ｐ": "p",
    "ｑ": "q",
    "ｒ": "r",
    "ｓ": "s",
    "ｔ": "t",
    "ｕ": "u",
    "ｖ": "v",
    "ｗ": "w",
    "ｘ": "x",
    "ｙ": "y",
    "ｚ": "z",
    "｛": "{",
    "｜": "|",
    "｝": "}",
    "～": "~",
}
QJ2BJ_TRANSFORM = str.maketrans("".join(QJ2BJ.keys()), "".join(QJ2BJ.values()), "")


# 2013 China National Standard: https://zh.wikipedia.org/wiki/通用规范汉字表, raw resources:
#   https://github.com/mozillazg/pinyin-data/blob/master/kMandarin_8105.txt with 8105 chinese chars in total
CN_CHARS_COMMON = (
    "一丁七万丈三上下不与丏丐丑专且丕世丘丙业丛东丝丞丢两严丧个丫中丰串临丸丹为主丽举"
    "乂乃久么义之乌乍乎乏乐乒乓乔乖乘乙乜九乞也习乡书乩买乱乳乸乾了予争事二亍于亏云互"
    "亓五井亘亚些亟亡亢交亥亦产亨亩享京亭亮亲亳亵亶亸亹人亿什仁仂仃仄仅仆仇仉今介仍从"
    "仑仓仔仕他仗付仙仝仞仟仡代令以仨仪仫们仰仲仳仵件价任份仿企伈伉伊伋伍伎伏伐休众优"
    "伙会伛伞伟传伢伣伤伥伦伧伪伫伭伯估伲伴伶伸伺似伽伾佁佃但位低住佐佑体何佖佗佘余佚"
    "佛作佝佞佟你佣佤佥佩佬佯佰佳佴佶佸佺佻佼佽佾使侁侂侃侄侈侉例侍侏侑侔侗侘供依侠侣"
    "侥侦侧侨侩侪侬侮侯侴侵侹便促俄俅俊俍俎俏俐俑俗俘俙俚俜保俞俟信俣俦俨俩俪俫俭修俯"
    "俱俳俵俶俸俺俾倌倍倏倒倓倔倕倘候倚倜倞借倡倥倦倧倨倩倪倬倭倮倴债倻值倾偁偃假偈偌"
    "偎偏偓偕做停偡健偬偭偰偲偶偷偻偾偿傀傃傅傈傉傍傒傕傣傥傧储傩催傲傺傻僇僎像僔僖僚"
    "僦僧僬僭僮僰僳僵僻儆儇儋儒儡儦儳儴儿兀允元兄充兆先光克免兑兔兕兖党兜兢入全八公六"
    "兮兰共关兴兵其具典兹养兼兽冀冁内冈冉册再冏冒冔冕冗写军农冠冢冤冥冬冮冯冰冱冲决况"
    "冶冷冻冼冽净凄准凇凉凋凌减凑凓凘凛凝几凡凤凫凭凯凰凳凶凸凹出击凼函凿刀刁刃分切刈"
    "刊刍刎刑划刖列刘则刚创初删判刨利别刬刭刮到刳制刷券刹刺刻刽刿剀剁剂剃剅削剋剌前剐"
    "剑剔剕剖剜剞剟剡剥剧剩剪副割剽剿劁劂劄劈劐劓力劝办功加务劢劣动助努劫劬劭励劲劳劼"
    "劾势勃勇勉勋勍勐勒勔勖勘勚募勠勤勰勺勾勿匀包匆匈匍匏匐匕化北匙匜匝匠匡匣匦匪匮匹"
    "区医匼匾匿十千卅升午卉半华协卑卒卓单卖南博卜卞卟占卡卢卣卤卦卧卫卬卮卯印危即却卵"
    "卷卸卺卿厂厄厅历厉压厌厍厕厖厘厚厝原厢厣厥厦厨厩厮去厾县叁参叆叇又叉及友双反发叔"
    "叕取受变叙叚叛叟叠口古句另叨叩只叫召叭叮可台叱史右叵叶号司叹叻叼叽吁吃各吆合吉吊"
    "同名后吏吐向吒吓吕吖吗君吝吞吟吠吡吣否吧吨吩含听吭吮启吱吲吴吵吸吹吻吼吽吾呀呃呆"
    "呇呈告呋呐呒呓呔呕呖呗员呙呛呜呢呣呤呦周呱呲味呵呶呷呸呻呼命咀咂咄咆咇咉咋和咍咎"
    "咏咐咒咔咕咖咙咚咛咝咡咣咤咥咦咧咨咩咪咫咬咯咱咳咴咸咺咻咽咿哀品哂哃哄哆哇哈哉哌"
    "响哎哏哐哑哒哓哔哕哗哙哚哝哞哟哢哥哦哧哨哩哪哭哮哱哲哳哺哼哽哿唁唆唇唉唏唐唑唔唛"
    "唝唠唢唣唤唧唪唬售唯唰唱唳唵唷唼唾唿啁啃啄商啉啊啐啕啖啜啡啤啥啦啧啪啫啬啭啮啰啴"
    "啵啶啷啸啻啼啾喀喁喂喃善喆喇喈喉喊喋喏喑喔喘喙喜喝喟喤喧喱喳喵喷喹喻喽喾嗄嗅嗉嗌"
    "嗍嗐嗑嗒嗓嗔嗖嗜嗝嗞嗟嗡嗣嗤嗥嗦嗨嗪嗫嗬嗯嗲嗳嗵嗷嗽嗾嘀嘁嘈嘉嘌嘎嘏嘘嘚嘛嘞嘟嘡"
    "嘣嘤嘧嘬嘭嘱嘲嘴嘶嘹嘻嘿噀噂噇噌噍噎噔噗噘噙噜噢噤器噩噪噫噬噱噶噻噼嚄嚅嚆嚎嚏嚓"
    "嚚嚣嚭嚯嚷嚼囊囔囚四回囟因囡团囤囫园困囱围囵囷囹固国图囿圃圄圆圈圉圊圌圐圙圜土圢"
    "圣在圩圪圫圬圭圮圯地圲圳圹场圻圾址坂均坉坊坋坌坍坎坏坐坑坒块坚坛坜坝坞坟坠坡坤坥"
    "坦坨坩坪坫坬坭坯坰坳坷坻坼坽垂垃垄垆垈型垌垍垎垏垒垓垕垙垚垛垞垟垠垡垢垣垤垦垧垩"
    "垫垭垮垯垱垲垴垵垸垺垾垿埂埃埆埇埋埌城埏埒埔埕埗埘埙埚埝域埠埤埪埫埭埯埴埵埸培基"
    "埼埽堂堃堆堇堉堋堌堍堎堐堑堕堙堞堠堡堤堧堨堪堰堲堵堼堽堾塄塅塆塌塍塑塔塘塝塞塥填"
    "塬塱塾墀墁境墅墈墉墐墒墓墕墘墙墚增墟墡墣墦墨墩墼壁壅壑壕壤士壬壮声壳壶壸壹处备复"
    "夏夐夔夕外夙多夜够夤夥大天太夫夬夭央夯失头夷夸夹夺夼奁奂奄奇奈奉奋奎奏契奓奔奕奖"
    "套奘奚奠奡奢奥奭女奴奶奸她好妁如妃妄妆妇妈妊妍妒妓妖妗妘妙妞妣妤妥妧妨妩妪妫妭妮"
    "妯妲妹妻妾姆姈姊始姐姑姒姓委姗姘姚姜姝姞姣姤姥姨姬姮姱姶姹姻姽姿娀威娃娄娅娆娇娈"
    "娉娌娑娓娘娜娟娠娣娥娩娱娲娴娵娶娼婀婆婉婊婌婍婕婘婚婞婠婢婤婧婪婫婳婴婵婶婷婺婻"
    "婼婿媂媄媆媒媓媖媚媛媞媪媭媱媲媳媵媸媾嫁嫂嫄嫉嫌嫒嫔嫕嫖嫘嫚嫜嫠嫡嫣嫦嫩嫪嫫嫭嫱"
    "嫽嬉嬖嬗嬛嬥嬬嬴嬷嬿孀孅子孑孓孔孕孖字存孙孚孛孜孝孟孢季孤孥学孩孪孬孰孱孳孵孺孽"
    "宁它宄宅宇守安宋完宏宓宕宗官宙定宛宜宝实宠审客宣室宥宦宧宪宫宬宰害宴宵家宸容宽宾"
    "宿寁寂寄寅密寇富寐寒寓寝寞察寡寤寥寨寮寰寸对寺寻导寿封射将尉尊小少尔尕尖尘尚尜尝"
    "尢尤尥尧尨尪尬就尴尸尹尺尻尼尽尾尿局屁层屃居屈屉届屋屎屏屐屑展屙属屠屡屣履屦屯山"
    "屹屺屼屾屿岁岂岈岊岌岍岐岑岔岖岗岘岙岚岛岜岞岠岢岣岨岩岫岬岭岱岳岵岷岸岽岿峁峂峃"
    "峄峋峒峗峘峙峛峡峣峤峥峦峧峨峪峭峰峱峻峿崀崁崂崃崄崆崇崌崎崒崔崖崚崛崞崟崡崤崦崧"
    "崩崭崮崴崶崽崾崿嵁嵅嵇嵊嵋嵌嵎嵖嵘嵚嵛嵝嵩嵫嵬嵯嵲嵴嶂嶅嶍嶒嶓嶙嶝嶟嶦嶲嶷巅巇巉"
    "巍川州巡巢工左巧巨巩巫差巯己已巳巴巷巽巾币市布帅帆师希帏帐帑帔帕帖帘帙帚帛帜帝帡"
    "带帧帨席帮帱帷常帻帼帽幂幄幅幌幔幕幖幛幞幡幢幪干平年并幸幺幻幼幽广庄庆庇床庋序庐"
    "庑库应底庖店庙庚府庞废庠庤庥度座庭庱庳庵庶康庸庹庼庾廆廉廊廋廑廒廓廖廙廛廨廪延廷"
    "建廿开弁异弃弄弆弇弈弊弋式弑弓引弗弘弛弟张弢弥弦弧弨弩弭弯弱弶弸弹强弼彀归当录彖"
    "彗彘彝彟形彤彦彧彩彪彬彭彰影彳彷役彻彼往征徂径待徇很徉徊律徐徒徕得徘徙徛徜御徨循"
    "徭微徵德徼徽心必忆忉忌忍忏忐忑忒忖志忘忙忝忞忠忡忤忧忪快忭忮忱忳念忸忺忻忽忾忿怀"
    "态怂怃怄怅怆怊怍怎怏怒怔怕怖怙怛怜思怠怡急怦性怨怩怪怫怯怵总怼怿恁恂恃恋恍恐恒恓"
    "恔恕恙恚恝恢恣恤恧恨恩恪恫恬恭息恰恳恶恸恹恺恻恼恽恿悃悄悆悈悉悌悍悒悔悖悚悛悝悟"
    "悠悢患悦您悫悬悭悯悰悱悲悴悸悻悼情惆惇惊惋惎惑惔惕惘惙惚惛惜惝惟惠惦惧惨惩惫惬惭"
    "惮惯惰想惴惶惹惺愀愁愃愆愈愉愍愎意愐愔愕愚感愠愣愤愦愧愫愭愿慆慈慊慌慎慑慕慝慢慥"
    "慧慨慬慭慰慵慷憋憎憔憕憙憧憨憩憬憭憷憺憾懂懈懊懋懑懒懔懦懵懿戆戈戊戋戌戍戎戏成我"
    "戒戕或戗战戚戛戟戡戢戣戤戥截戬戭戮戳戴户戽戾房所扁扂扃扅扆扇扈扉扊手才扎扑扒打扔"
    "托扛扞扣扦执扩扪扫扬扭扮扯扰扳扶批扺扼扽找承技抃抄抉把抑抒抓抔投抖抗折抚抛抟抠抡"
    "抢护报抨披抬抱抵抹抻押抽抿拂拃拄担拆拇拈拉拊拌拍拎拐拒拓拔拖拗拘拙招拜拟拢拣拤拥"
    "拦拧拨择括拭拮拯拱拳拴拶拷拼拽拾拿持挂指挈按挎挑挓挖挚挛挝挞挟挠挡挣挤挥挦挨挪挫"
    "振挲挹挺挽捂捃捅捆捉捋捌捍捎捏捐捕捞损捡换捣捧捩捭据捯捶捷捺捻捽掀掂掇授掉掊掌掎"
    "掏掐排掖掘掞掠探掣接控推掩措掬掭掮掰掳掴掷掸掺掼掾揄揆揉揍描提插揕揖揠握揣揩揪揭"
    "揳援揶揸揽揿搀搁搂搅搋搌搏搐搒搓搔搛搜搞搠搡搦搪搬搭搴携搽摁摄摅摆摇摈摊摏摒摔摘"
    "摛摞摧摩摭摴摸摹摽撂撄撅撇撑撒撕撖撙撞撤撩撬播撮撰撵撷撸撺撼擀擂擅操擎擐擒擘擞擢"
    "擤擦擿攀攉攒攘攥攫攮支收攸改攻攽放政故效敉敌敏救敔敕敖教敛敝敞敢散敦敩敫敬数敲整"
    "敷文斋斌斐斑斓斗料斛斜斝斟斠斡斤斥斧斩斫断斯新斶方於施旁旃旄旅旆旋旌旎族旐旒旖旗"
    "旞无既日旦旧旨早旬旭旮旯旰旱旴旵时旷旸旺旻旿昀昂昃昄昆昇昈昉昊昌明昏昒易昔昕昙昝"
    "星映昡昣昤春昧昨昪昫昭是昱昳昴昵昶昺昼昽显晁晃晅晊晋晌晏晐晒晓晔晕晖晗晙晚晞晟晡"
    "晢晤晦晨晪晫普景晰晱晴晶晷智晾暂暄暅暇暌暑暕暖暗暝暧暨暮暲暴暵暶暹暾暿曈曌曙曛曜"
    "曝曦曩曰曲曳更曷曹曼曾替最月有朋服朏朐朓朔朕朗望朝期朦木未末本札术朱朳朴朵朸机朽"
    "杀杂权杄杆杈杉杌李杏材村杓杕杖杙杜杞束杠条来杧杨杩杪杭杯杰杲杳杵杷杻杼松板极构枅"
    "枇枉枋枍析枕林枘枚果枝枞枢枣枥枧枨枪枫枭枯枰枲枳枵架枷枸枹柁柃柄柈柊柏某柑柒染柔"
    "柖柘柙柚柜柝柞柠柢查柩柬柯柰柱柳柴柷柽柿栀栅标栈栉栊栋栌栎栏栐树栒栓栖栗栝栟校栩"
    "株栲栳栴样核根栻格栽栾桀桁桂桃桄桅框案桉桊桌桎桐桑桓桔桕桠桡桢档桤桥桦桧桨桩桫桯"
    "桲桴桶桷桹梁梃梅梆梌梏梓梗梠梢梣梦梧梨梭梯械梳梴梵梼梽梾梿检棁棂棉棋棍棐棒棓棕棘"
    "棚棠棣棤棨棪棫棬森棰棱棵棹棺棻棼棽椀椁椅椆椋植椎椐椑椒椓椟椠椤椪椭椰椴椸椹椽椿楂"
    "楒楔楗楙楚楝楞楠楣楦楩楪楫楮楯楷楸楹楼概榃榄榅榆榇榈榉榍榑榔榕榖榛榜榧榨榫榭榰榱"
    "榴榷榻槁槃槊槌槎槐槔槚槛槜槟槠槭槱槲槽槿樊樗樘樟模樨横樯樱樵樽樾橄橇橐橑橘橙橛橞"
    "橡橥橦橱橹橼檀檄檎檐檑檗檞檠檩檫檬櫆欂欠次欢欣欤欧欲欸欹欺欻款歃歅歆歇歉歌歙止正"
    "此步武歧歪歹死歼殁殂殃殄殆殇殉殊残殍殒殓殖殚殛殡殣殪殳殴段殷殿毁毂毅毋毌母每毐毒"
    "毓比毕毖毗毙毛毡毪毫毯毳毵毹毽氅氆氇氍氏氐民氓气氕氖氘氙氚氛氟氡氢氤氦氧氨氩氪氮"
    "氯氰氲水永氾氿汀汁求汆汇汈汉汊汋汐汔汕汗汛汜汝汞江池污汤汧汨汩汪汫汭汰汲汴汶汹汽"
    "汾沁沂沃沄沅沆沇沈沉沌沏沐沓沔沘沙沚沛沟没沣沤沥沦沧沨沩沪沫沭沮沱河沸油沺治沼沽"
    "沾沿泂泃泄泅泇泉泊泌泐泓泔法泖泗泙泚泛泜泞泠泡波泣泥注泪泫泮泯泰泱泳泵泷泸泺泻泼"
    "泽泾洁洄洇洈洋洌洎洑洒洓洗洘洙洚洛洞洢洣津洧洨洪洫洭洮洱洲洳洴洵洸洹洺活洼洽派洿"
    "流浃浅浆浇浈浉浊测浍济浏浐浑浒浓浔浕浙浚浛浜浞浟浠浡浣浥浦浩浪浬浭浮浯浰浲浴海浸"
    "浼涂涄涅消涉涌涍涎涐涑涓涔涕涘涛涝涞涟涠涡涢涣涤润涧涨涩涪涫涮涯液涴涵涸涿淀淄淅"
    "淆淇淋淌淏淑淖淘淙淜淝淞淟淠淡淤淦淫淬淮淯深淳淴混淹添淼清渊渌渍渎渐渑渔渗渚渝渟"
    "渠渡渣渤渥温渫渭港渰渲渴游渺渼湃湄湉湍湎湑湓湔湖湘湛湜湝湟湣湫湮湲湴湾湿溁溃溅溆"
    "溇溉溍溏源溘溚溜溞溟溠溢溥溦溧溪溯溱溲溴溵溶溷溹溺溻溽滁滂滃滆滇滉滋滍滏滑滓滔滕"
    "滗滘滚滞滟滠满滢滤滥滦滧滨滩滪滫滴滹漂漆漈漉漋漏漓演漕漖漠漤漦漩漪漫漭漯漱漳漴漶"
    "漷漹漻漼漾潆潇潋潍潏潖潘潜潞潟潢潦潩潭潮潲潴潵潸潺潼潽潾澂澄澈澉澌澍澎澛澜澡澥澧"
    "澪澭澳澴澶澹澼澽激濂濉濋濑濒濞濠濡濩濮濯瀌瀍瀑瀔瀚瀛瀣瀱瀵瀹瀼灈灌灏灞火灭灯灰灵"
    "灶灸灼灾灿炀炅炆炉炊炌炎炒炔炕炖炘炙炜炝炟炣炫炬炭炮炯炱炳炷炸点炻炼炽烀烁烂烃烈"
    "烊烔烘烙烛烜烝烟烠烤烦烧烨烩烫烬热烯烶烷烹烺烻烽焆焉焊焌焐焓焕焖焗焘焙焚焜焞焦焯"
    "焰焱然煁煃煅煊煋煌煎煓煜煞煟煤煦照煨煮煲煳煴煸煺煽熄熇熊熏熔熘熙熛熜熟熠熥熨熬熵"
    "熹熻燃燊燋燎燏燔燕燚燠燥燧燮燹爆爇爔爚爝爟爨爪爬爰爱爵父爷爸爹爻爽爿牁牂片版牌牍"
    "牒牖牙牚牛牝牟牡牢牤牥牦牧物牮牯牲牵特牺牻牾牿犀犁犄犇犊犋犍犏犒犟犨犬犯犰犴状犷"
    "犸犹狁狂狃狄狈狉狍狎狐狒狗狙狝狞狠狡狨狩独狭狮狯狰狱狲狳狴狷狸狺狻狼猁猃猄猇猊猎"
    "猕猖猗猛猜猝猞猡猢猥猩猪猫猬献猯猰猱猴猷猹猺猾猿獍獐獒獗獠獬獭獯獴獾玃玄率玉王玎"
    "玑玒玓玕玖玘玙玚玛玞玟玠玡玢玤玥玦玩玫玭玮环现玱玲玳玶玷玹玺玻玼玿珀珂珅珇珈珉珊"
    "珋珌珍珏珐珑珒珕珖珙珛珝珞珠珢珣珥珦珧珩珪珫班珰珲珵珷珸珹珺珽琀球琄琅理琇琈琉琊"
    "琎琏琐琔琚琛琟琡琢琤琥琦琨琪琫琬琭琮琯琰琲琳琴琵琶琼瑀瑁瑂瑃瑄瑅瑆瑑瑓瑔瑕瑖瑗瑙"
    "瑚瑛瑜瑝瑞瑟瑢瑧瑨瑬瑭瑰瑱瑳瑶瑷瑾璀璁璃璆璇璈璋璎璐璒璘璜璞璟璠璥璧璨璩璪璬璮璱"
    "璲璺瓀瓒瓖瓘瓜瓞瓠瓢瓣瓤瓦瓮瓯瓴瓶瓷瓻瓿甄甍甏甑甓甗甘甚甜生甡甥甦用甩甪甫甬甭甯"
    "田由甲申电男甸町画甾畀畅畈畋界畎畏畔畖留畚畛畜畤略畦番畬畯畲畴畸畹畿疁疃疆疍疏疐"
    "疑疔疖疗疙疚疝疟疠疡疢疣疤疥疫疬疭疮疯疰疱疲疳疴疵疸疹疼疽疾痂痃痄病症痈痉痊痍痒"
    "痓痔痕痘痛痞痢痣痤痦痧痨痪痫痰痱痴痹痼痿瘀瘁瘃瘅瘆瘊瘌瘐瘕瘗瘘瘙瘛瘟瘠瘢瘤瘥瘦瘩"
    "瘪瘫瘭瘰瘳瘴瘵瘸瘼瘾瘿癀癃癌癍癔癖癗癜癞癣癫癯癸登白百癿皂的皆皇皈皋皎皑皓皕皖皙"
    "皛皞皤皦皭皮皱皲皴皿盂盅盆盈盉益盍盎盏盐监盒盔盖盗盘盛盟盥盦目盯盱盲直盷相盹盼盾"
    "省眄眇眈眉眊看眍眙眚真眠眢眦眨眩眬眭眯眵眶眷眸眺眼着睁睃睄睇睎睐睑睚睛睡睢督睥睦"
    "睨睫睬睹睽睾睿瞀瞄瞅瞋瞌瞍瞎瞑瞒瞟瞠瞢瞥瞧瞩瞪瞫瞬瞭瞰瞳瞵瞻瞽瞿矍矗矛矜矞矢矣知"
    "矧矩矫矬短矮矰石矶矸矻矼矾矿砀码砂砄砆砉砌砍砑砒研砖砗砘砚砜砝砟砠砣砥砧砫砬砭砮"
    "砰破砵砷砸砹砺砻砼砾础硁硅硇硊硌硍硎硐硒硔硕硖硗硙硚硝硪硫硬硭确硼硿碃碇碈碉碌碍"
    "碎碏碑碓碗碘碚碛碜碟碡碣碥碧碨碰碱碲碳碴碶碹碾磁磅磉磊磋磏磐磔磕磙磜磡磨磬磲磴磷"
    "磹磻礁礅礌礓礞礴礵示礼社祀祁祃祆祇祈祉祊祋祎祏祐祓祕祖祗祚祛祜祝神祟祠祢祥祧票祭"
    "祯祲祷祸祺祼祾禀禁禄禅禊禋福禒禔禘禚禛禤禧禳禹禺离禽禾秀私秃秆秉秋种科秒秕秘租秣"
    "秤秦秧秩秫秬秭积称秸移秽秾稀稂稃稆程稌稍税稑稔稗稙稚稞稠稣稳稷稹稻稼稽稿穄穆穑穗"
    "穙穜穟穰穴究穷穸穹空穿窀突窃窄窅窈窊窍窎窑窒窕窖窗窘窜窝窟窠窣窥窦窨窬窭窳窸窿立"
    "竑竖竘站竞竟章竣童竦竫竭端竹竺竽竿笃笄笆笈笊笋笏笑笔笕笙笛笞笠笤笥符笨笪笫第笮笯"
    "笱笳笸笺笼笾筀筅筇等筋筌筏筐筑筒答策筘筚筛筜筝筠筢筤筥筦筮筱筲筵筶筷筹筻筼签简箅"
    "箍箐箓箔箕箖算箜管箢箦箧箨箩箪箫箬箭箱箴箸篁篆篇篌篑篓篙篚篝篡篥篦篪篮篯篱篷篼篾"
    "簃簇簉簋簌簏簕簖簝簟簠簧簪簰簸簿籀籁籍籥米籴类籼籽粉粑粒粕粗粘粜粝粞粟粢粤粥粪粮"
    "粱粲粳粹粼粽精粿糁糅糇糈糊糌糍糒糕糖糗糙糜糟糠糨糯糵系紊素索紧紫累絜絮絷綦綮縠縢"
    "縻繁繄繇纂纛纠纡红纣纤纥约级纨纩纪纫纬纭纮纯纰纱纲纳纴纵纶纷纸纹纺纻纼纽纾线绀绁"
    "绂练组绅细织终绉绊绋绌绍绎经绐绑绒结绔绕绖绗绘给绚绛络绝绞统绠绡绢绣绤绥绦继绨绩"
    "绪绫续绮绯绰绱绲绳维绵绶绷绸绹绺绻综绽绾绿缀缁缂缃缄缅缆缇缈缉缊缌缎缐缑缒缓缔缕"
    "编缗缘缙缚缛缜缝缞缟缠缡缢缣缤缥缦缧缨缩缪缫缬缭缮缯缰缱缲缳缴缵缶缸缺罂罄罅罍罐"
    "网罔罕罗罘罚罟罡罢罨罩罪置罱署罴罶罹罽罾羁羊羌美羑羓羔羕羖羚羝羞羟羡群羧羯羰羱羲"
    "羸羹羼羽羿翀翁翂翃翅翈翊翌翎翔翕翘翙翚翛翟翠翡翥翦翩翮翯翰翱翳翷翻翼翾耀老考耄者"
    "耆耇耋而耍耏耐耑耒耔耕耖耗耘耙耜耠耢耤耥耦耧耨耩耪耰耱耳耵耶耷耸耻耽耿聂聃聆聊聋"
    "职聍聒联聘聚聩聪聱聿肃肄肆肇肉肋肌肓肖肘肚肛肝肟肠股肢肤肥肩肪肫肭肮肯肱育肴肷肸"
    "肺肼肽肾肿胀胁胂胃胄胆胈背胍胎胖胗胙胚胛胜胝胞胠胡胣胤胥胧胨胩胪胫胬胭胯胰胱胲胳"
    "胴胶胸胺胼能脂脆脉脊脍脎脏脐脑脒脓脔脖脘脚脞脟脩脬脯脱脲脶脸脾脿腆腈腊腋腌腐腑腒"
    "腓腔腕腘腙腚腠腥腧腨腩腭腮腯腰腱腴腹腺腻腼腽腾腿膀膂膈膊膏膑膘膙膛膜膝膦膨膳膺膻"
    "臀臂臃臆臊臌臑臜臣臧自臬臭至致臻臼臾舀舁舂舄舅舆舌舍舐舒舔舛舜舞舟舠舢舣舥航舫般"
    "舭舯舰舱舲舳舴舵舶舷舸船舻舾艄艅艇艉艋艎艏艘艚艟艨艮良艰色艳艴艺艽艾艿节芃芄芈芊"
    "芋芍芎芏芑芒芗芘芙芜芝芟芠芡芣芤芥芦芨芩芪芫芬芭芮芯芰花芳芴芷芸芹芼芽芾苁苄苇苈"
    "苉苊苋苌苍苎苏苑苒苓苔苕苗苘苛苜苞苟苠苡苣苤若苦苧苫苯英苴苷苹苻苾茀茁茂范茄茅茆"
    "茈茉茋茌茎茏茑茓茔茕茗茚茛茜茝茧茨茫茬茭茯茱茳茴茵茶茸茹茺茼茽荀荁荃荄荆荇草荏荐"
    "荑荒荓荔荖荙荚荛荜荞荟荠荡荣荤荥荦荧荨荩荪荫荬荭荮药荷荸荻荼荽莅莆莉莎莒莓莘莙莛"
    "莜莝莞莠莨莩莪莫莰莱莲莳莴莶获莸莹莺莼莽莿菀菁菂菅菇菉菊菌菍菏菔菖菘菜菝菟菠菡菥"
    "菩菪菰菱菲菹菼菽萁萃萄萆萋萌萍萎萏萑萘萚萜萝萣萤营萦萧萨萩萱萳萸萹萼落葆葎葑葖著"
    "葙葚葛葜葡董葩葫葬葭葰葱葳葴葵葶葸葺蒂蒄蒇蒈蒉蒋蒌蒎蒐蒗蒙蒜蒟蒡蒨蒯蒱蒲蒴蒸蒹蒺"
    "蒻蒽蒿蓁蓂蓄蓇蓉蓊蓍蓏蓐蓑蓓蓖蓝蓟蓠蓢蓣蓥蓦蓬蓰蓼蓿蔀蔃蔈蔊蔌蔑蔓蔗蔚蔟蔡蔫蔬蔷"
    "蔸蔹蔺蔻蔼蔽蕃蕈蕉蕊蕖蕗蕙蕞蕤蕨蕰蕲蕴蕹蕺蕻蕾薁薄薅薇薏薛薜薢薤薨薪薮薯薰薳薷薸"
    "薹薿藁藉藏藐藓藕藜藟藠藤藦藨藩藻藿蘅蘑蘖蘘蘧蘩蘸蘼虎虏虐虑虒虓虔虚虞虢虤虫虬虮虱"
    "虷虸虹虺虻虼虽虾虿蚀蚁蚂蚄蚆蚊蚋蚌蚍蚓蚕蚜蚝蚣蚤蚧蚨蚩蚪蚬蚯蚰蚱蚲蚴蚶蚺蛀蛃蛄蛆"
    "蛇蛉蛊蛋蛎蛏蛐蛑蛔蛘蛙蛛蛞蛟蛤蛩蛭蛮蛰蛱蛲蛳蛴蛸蛹蛾蜀蜂蜃蜇蜈蜉蜊蜍蜎蜐蜒蜓蜕蜗"
    "蜘蜚蜜蜞蜡蜢蜣蜥蜩蜮蜱蜴蜷蜻蜾蜿蝇蝈蝉蝌蝎蝓蝗蝘蝙蝠蝣蝤蝥蝮蝰蝲蝴蝶蝻蝼蝽蝾螂螃"
    "螅螈螋融螗螟螠螣螨螫螬螭螯螱螳螵螺螽蟀蟆蟊蟋蟏蟑蟒蟛蟠蟥蟪蟫蟮蟹蟾蠃蠊蠋蠓蠕蠖蠡"
    "蠢蠲蠹蠼血衃衄衅行衍衎衒衔街衙衠衡衢衣补表衩衫衬衮衰衲衷衽衾衿袁袂袄袅袆袈袋袍袒"
    "袖袗袜袢袤袪被袭袯袱袷袼裁裂装裆裈裉裎裒裔裕裘裙裛裟裢裣裤裥裨裰裱裳裴裸裹裼裾褂"
    "褊褐褒褓褕褙褚褛褟褡褥褪褫褯褰褴褶襁襄襕襚襜襞襟襦襫襻西要覃覆见观觃规觅视觇览觉"
    "觊觋觌觎觏觐觑角觖觚觜觞觟解觥触觫觭觯觱觳觿言訄訇訚訾詈詟詹誉誊誓謇警譬计订讣认"
    "讥讦讧讨让讪讫训议讯记讱讲讳讴讵讶讷许讹论讻讼讽设访诀证诂诃评诅识诇诈诉诊诋诌词"
    "诎诏诐译诒诓诔试诖诗诘诙诚诛诜话诞诟诠诡询诣诤该详诧诨诩诫诬语诮误诰诱诲诳说诵请"
    "诸诹诺读诼诽课诿谀谁谂调谄谅谆谇谈谊谋谌谍谎谏谐谑谒谓谔谕谖谗谙谚谛谜谝谞谟谠谡"
    "谢谣谤谥谦谧谨谩谪谫谬谭谮谯谰谱谲谳谴谵谶谷谼谿豁豆豇豉豌豕豚象豢豨豪豫豮豳豸豹"
    "豺貂貅貆貉貊貌貔貘贝贞负贡财责贤败账货质贩贪贫贬购贮贯贰贱贲贳贴贵贶贷贸费贺贻贼"
    "贽贾贿赀赁赂赃资赅赆赇赈赉赊赋赌赍赎赏赐赑赒赓赔赕赖赗赘赙赚赛赜赝赞赟赠赡赢赣赤"
    "赦赧赪赫赭走赳赴赵赶起趁趄超越趋趑趔趟趣趯趱足趴趵趸趺趼趾趿跂跃跄跆跋跌跎跏跐跑"
    "跖跗跚跛距跞跟跣跤跨跪跬路跱跳践跶跷跸跹跺跻跽踅踉踊踌踏踒踔踝踞踟踢踣踦踩踪踬踮"
    "踯踱踵踶踹踺踽蹀蹁蹂蹄蹅蹇蹈蹉蹊蹋蹐蹑蹒蹙蹚蹜蹢蹦蹩蹬蹭蹯蹰蹲蹴蹶蹼蹽蹾蹿躁躅躇"
    "躏躐躔躜躞身躬躯躲躺车轧轨轩轪轫转轭轮软轰轱轲轳轴轵轶轷轸轹轺轻轼载轾轿辀辁辂较"
    "辄辅辆辇辈辉辊辋辌辍辎辏辐辑辒输辔辕辖辗辘辙辚辛辜辞辟辣辨辩辫辰辱边辽达辿迁迂迄"
    "迅过迈迎运近迓返迕还这进远违连迟迢迤迥迦迨迩迪迫迭迮述迳迷迸迹迺追退送适逃逄逅逆"
    "选逊逋逍透逐逑递途逖逗通逛逝逞速造逡逢逦逭逮逯逴逵逶逸逻逼逾遁遂遄遆遇遍遏遐遑遒"
    "道遗遘遛遢遣遥遨遭遮遴遵遹遽避邀邂邃邈邋邑邓邕邗邘邙邛邝邠邡邢那邦邨邪邬邮邯邰邱"
    "邲邳邴邵邶邸邹邺邻邽邾邿郁郃郄郅郇郈郊郎郏郐郑郓郗郚郛郜郝郡郢郤郦郧部郪郫郭郯郴"
    "郸都郾郿鄀鄂鄃鄄鄅鄌鄑鄗鄘鄙鄚鄜鄞鄠鄢鄣鄫鄯鄱鄹酂酃酅酆酉酊酋酌配酎酏酐酒酗酚酝"
    "酞酡酢酣酤酥酦酩酪酬酮酯酰酱酲酴酵酶酷酸酹酺酽酾酿醅醇醉醋醌醍醐醑醒醚醛醢醨醪醭"
    "醮醯醴醵醺醾采釉释里重野量釐金釜鉴銎銮鋆鋈錾鍪鎏鏊鏖鐾鑫钆钇针钉钊钋钌钍钎钏钐钒"
    "钓钔钕钖钗钘钙钚钛钜钝钞钟钠钡钢钣钤钥钦钧钨钩钪钫钬钭钮钯钰钱钲钳钴钵钷钹钺钻钼"
    "钽钾钿铀铁铂铃铄铅铆铈铉铊铋铌铍铎铏铐铑铒铕铖铗铘铙铚铛铜铝铞铟铠铡铢铣铤铥铧铨"
    "铩铪铫铬铭铮铯铰铱铲铳铴铵银铷铸铹铺铻铼铽链铿销锁锂锃锄锅锆锇锈锉锊锋锌锍锎锏锐"
    "锑锒锓锔锕锖锗锘错锚锛锜锝锞锟锡锢锣锤锥锦锧锨锩锪锫锬锭键锯锰锱锲锳锴锵锶锷锸锹"
    "锺锻锼锽锾锿镀镁镂镃镄镅镆镇镈镉镊镋镌镍镎镏镐镑镒镓镔镕镖镗镘镚镛镜镝镞镠镡镢镣"
    "镤镥镦镧镨镩镪镫镬镭镮镯镰镱镲镳镴镵镶长门闩闪闫闭问闯闰闱闲闳间闵闶闷闸闹闺闻闼"
    "闽闾闿阀阁阂阃阄阅阆阇阈阉阊阋阌阍阎阏阐阑阒阔阕阖阗阘阙阚阜队阡阪阮阱防阳阴阵阶"
    "阻阼阽阿陀陂附际陆陇陈陉陋陌降陎限陑陔陕陛陞陟陡院除陧陨险陪陬陲陴陵陶陷隃隅隆隈"
    "隋隍随隐隔隗隘隙障隧隩隰隳隶隹隺隼隽难雀雁雄雅集雇雉雊雌雍雎雏雒雕雠雨雩雪雯雱雳"
    "零雷雹雾需霁霄霅霆震霈霉霍霎霏霓霖霜霞霨霪霭霰露霸霹霾青靓靖静靛非靠靡面靥革靬靰"
    "靳靴靶靸靺靼靽靿鞁鞅鞋鞍鞑鞒鞔鞘鞠鞡鞣鞧鞨鞫鞬鞭鞮鞯鞲鞳鞴韂韦韧韨韩韪韫韬韭音韵"
    "韶页顶顷顸项顺须顼顽顾顿颀颁颂颃预颅领颇颈颉颊颋颌颍颎颏颐频颓颔颖颗题颙颚颛颜额"
    "颞颟颠颡颢颤颥颦颧风飏飐飑飒飓飔飕飗飘飙飞食飧飨餍餐餮饔饕饥饧饨饩饪饫饬饭饮饯饰"
    "饱饲饳饴饵饶饷饸饹饺饻饼饽饿馁馃馄馅馆馇馈馉馊馋馌馍馏馐馑馒馓馔馕首馗馘香馝馞馥"
    "馧馨马驭驮驯驰驱驲驳驴驵驶驷驸驹驺驻驼驽驾驿骀骁骂骃骄骅骆骇骈骉骊骋验骍骎骏骐骑"
    "骒骓骕骖骗骘骙骚骛骜骝骞骟骠骡骢骣骤骥骦骧骨骰骱骶骷骸骺骼髀髁髂髃髅髋髌髎髑髓高"
    "髡髢髦髫髭髯髹髻髽鬃鬈鬏鬒鬓鬘鬟鬣鬯鬲鬶鬷鬻鬼魁魂魃魄魅魆魇魈魉魋魍魏魑魔鱼鱽鱾"
    "鱿鲀鲁鲂鲃鲅鲆鲇鲈鲉鲊鲋鲌鲍鲎鲏鲐鲑鲒鲔鲕鲖鲗鲘鲙鲚鲛鲜鲝鲞鲟鲠鲡鲢鲣鲤鲥鲦鲧鲨"
    "鲩鲪鲫鲬鲭鲮鲯鲰鲱鲲鲳鲴鲵鲷鲸鲹鲺鲻鲼鲽鲾鲿鳀鳁鳂鳃鳄鳅鳇鳈鳉鳊鳌鳍鳎鳏鳐鳑鳒鳓"
    "鳔鳕鳖鳗鳘鳙鳚鳛鳜鳝鳞鳟鳠鳡鳢鳣鳤鸟鸠鸡鸢鸣鸤鸥鸦鸧鸨鸩鸪鸫鸬鸭鸮鸯鸰鸱鸲鸳鸵鸶"
    "鸷鸸鸹鸺鸻鸼鸽鸾鸿鹀鹁鹂鹃鹄鹅鹆鹇鹈鹉鹊鹋鹌鹍鹎鹏鹐鹑鹒鹔鹕鹖鹗鹘鹙鹚鹛鹜鹝鹞鹟"
    "鹠鹡鹢鹣鹤鹦鹧鹨鹩鹪鹫鹬鹭鹮鹯鹰鹱鹲鹳鹴鹾鹿麀麂麇麈麋麑麒麓麖麝麟麦麸麹麻麽麾黄"
    "黇黉黍黎黏黑黔默黛黜黝黟黠黡黢黥黧黩黪黯黹黻黼黾鼋鼍鼎鼐鼒鼓鼗鼙鼠鼢鼩鼫鼬鼯鼱鼷"
    "鼹鼻鼽鼾齁齇齉齐齑齿龀龁龂龃龄龅龆龇龈龉龊龋龌龙龚龛龟龠龢鿍鿎鿏㑇㑊㕮㘎㙍㙘㙦㛃"
    "㛚㛹㟃㠇㠓㤘㥄㧐㧑㧟㫰㬊㬎㬚㭎㭕㮾㰀㳇㳘㳚㴔㵐㶲㸆㸌㺄㻬㽏㿠䁖䂮䃅䃎䅟䌹䎃䎖䏝䏡"
    "䏲䐃䓖䓛䓨䓫䓬䗖䗛䗪䗴䜣䝙䢺䢼䣘䥽䦃䲟䲠䲢䴓䴔䴕䴖䴗䴘䴙䶮𠅤𠙶𠳐𡎚𡐓𣗋𣲗𣲘𣸣𤧛𤩽"
    "𤫉𥔲𥕢𥖨𥻗𦈡𦒍𦙶𦝼𦭜𦰡𧿹𨐈𨙸𨚕𨟠𨭉𨱇𨱏𨱑𨱔𨺙𩽾𩾃𩾌𪟝𪣻𪤗𪨰𪨶𪩘𪾢𫄧𫄨𫄷𫄸𫇭𫌀𫍣𫍯"
    "𫍲𫍽𫐄𫐐𫐓𫑡𫓧𫓯𫓶𫓹𫔍𫔎𫔶𫖮𫖯𫖳𫗧𫗴𫘜𫘝𫘦𫘧𫘨𫘪𫘬𫚕𫚖𫚭𫛭𫞩𫟅𫟦𫟹𫟼𫠆𫠊𫠜𫢸𫫇𫭟"
    "𫭢𫭼𫮃𫰛𫵷𫶇𫷷𫸩𬀩𬀪𬂩𬃊𬇕𬇙𬇹𬉼𬊈𬊤𬌗𬍛𬍡𬍤𬒈𬒔𬒗𬕂𬘓𬘘𬘡𬘩𬘫𬘬𬘭𬘯𬙂𬙊𬙋𬜬𬜯𬞟"
    "𬟁𬟽𬣙𬣞𬣡𬣳𬤇𬤊𬤝𬨂𬨎𬩽𬪩𬬩𬬭𬬮𬬱𬬸𬬹𬬻𬬿𬭁𬭊𬭎𬭚𬭛𬭤𬭩𬭬𬭯𬭳𬭶𬭸𬭼𬮱𬮿𬯀𬯎𬱖𬱟"
    "𬳵𬳶𬳽𬳿𬴂𬴃𬴊𬶋𬶍𬶏𬶐𬶟𬶠𬶨𬶭𬶮𬷕𬸘𬸚𬸣𬸦𬸪𬹼𬺈𬺓"
)
CN_CHARS_EXT = "吶诶屌囧飚屄"

CN_CHARS = CN_CHARS_COMMON + CN_CHARS_EXT
IN_CH_CHARS = {c: True for c in CN_CHARS}

EN_CHARS = string.ascii_letters + string.digits
IN_EN_CHARS = {c: True for c in EN_CHARS}

VALID_CHARS = CN_CHARS + EN_CHARS + " "
IN_VALID_CHARS = {c: True for c in VALID_CHARS}


# ================================================================================ #
#                                    basic class
# ================================================================================ #
class ChineseChar(object):
    """
    中文字符
    每个字符对应简体和繁体,
    e.g. 简体 = '负', 繁体 = '負'
    转换时可转换为简体或繁体
    """

    def __init__(self, simplified, traditional):
        self.simplified = simplified
        self.traditional = traditional
        # self.__repr__ = self.__str__

    def __str__(self):
        return self.simplified or self.traditional or None

    def __repr__(self):
        return self.__str__()


class ChineseNumberUnit(ChineseChar):
    """
    中文数字/数位字符
    每个字符除繁简体外还有一个额外的大写字符
    e.g. '陆' 和 '陸'
    """

    def __init__(self, power, simplified, traditional, big_s, big_t):
        super(ChineseNumberUnit, self).__init__(simplified, traditional)
        self.power = power
        self.big_s = big_s
        self.big_t = big_t

    def __str__(self):
        return "10^{}".format(self.power)

    @classmethod
    def create(cls, index, value, numbering_type=NUMBERING_TYPES[1], small_unit=False):
        if small_unit:
            return ChineseNumberUnit(
                power=index + 1, simplified=value[0], traditional=value[1], big_s=value[1], big_t=value[1]
            )
        elif numbering_type == NUMBERING_TYPES[0]:
            return ChineseNumberUnit(
                power=index + 8, simplified=value[0], traditional=value[1], big_s=value[0], big_t=value[1]
            )
        elif numbering_type == NUMBERING_TYPES[1]:
            return ChineseNumberUnit(
                power=(index + 2) * 4, simplified=value[0], traditional=value[1], big_s=value[0], big_t=value[1]
            )
        elif numbering_type == NUMBERING_TYPES[2]:
            return ChineseNumberUnit(
                power=pow(2, index + 3), simplified=value[0], traditional=value[1], big_s=value[0], big_t=value[1]
            )
        else:
            raise ValueError("Counting type should be in {0} ({1} provided).".format(NUMBERING_TYPES, numbering_type))


class ChineseNumberDigit(ChineseChar):
    """
    中文数字字符
    """

    def __init__(self, value, simplified, traditional, big_s, big_t, alt_s=None, alt_t=None):
        super(ChineseNumberDigit, self).__init__(simplified, traditional)
        self.value = value
        self.big_s = big_s
        self.big_t = big_t
        self.alt_s = alt_s
        self.alt_t = alt_t

    def __str__(self):
        return str(self.value)

    @classmethod
    def create(cls, i, v):
        return ChineseNumberDigit(i, v[0], v[1], v[2], v[3])


class ChineseMath(ChineseChar):
    """
    中文数位字符
    """

    def __init__(self, simplified, traditional, symbol, expression=None):
        super(ChineseMath, self).__init__(simplified, traditional)
        self.symbol = symbol
        self.expression = expression
        self.big_s = simplified
        self.big_t = traditional


CC, CNU, CND, CM = ChineseChar, ChineseNumberUnit, ChineseNumberDigit, ChineseMath


class NumberSystem(object):
    """
    中文数字系统
    """

    pass


class MathSymbol(object):
    """
    用于中文数字系统的数学符号 (繁/简体), e.g.
    positive = ['正', '正']
    negative = ['负', '負']
    point = ['点', '點']
    """

    def __init__(self, positive, negative, point):
        self.positive = positive
        self.negative = negative
        self.point = point

    def __iter__(self):
        for v in self.__dict__.values():
            yield v


# class OtherSymbol(object):
#     """
#     其他符号
#     """
#
#     def __init__(self, sil):
#         self.sil = sil
#
#     def __iter__(self):
#         for v in self.__dict__.values():
#             yield v


# ================================================================================ #
#                                    basic utils
# ================================================================================ #
def create_system(numbering_type=NUMBERING_TYPES[1]):
    """
    根据数字系统类型返回创建相应的数字系统，默认为 mid
    NUMBERING_TYPES = ['low', 'mid', 'high']: 中文数字系统类型
        low:  '兆' = '亿' * '十' = $10^{9}$,  '京' = '兆' * '十', etc.
        mid:  '兆' = '亿' * '万' = $10^{12}$, '京' = '兆' * '万', etc.
        high: '兆' = '亿' * '亿' = $10^{16}$, '京' = '兆' * '兆', etc.
    返回对应的数字系统
    """

    # chinese number units of '亿' and larger
    all_larger_units = zip(LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED, LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL)
    larger_units = [CNU.create(i, v, numbering_type, False) for i, v in enumerate(all_larger_units)]
    # chinese number units of '十, 百, 千, 万'
    all_smaller_units = zip(SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED, SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL)
    smaller_units = [CNU.create(i, v, small_unit=True) for i, v in enumerate(all_smaller_units)]
    # digis
    chinese_digis = zip(CHINESE_DIGIS, CHINESE_DIGIS, BIG_CHINESE_DIGIS_SIMPLIFIED, BIG_CHINESE_DIGIS_TRADITIONAL)
    digits = [CND.create(i, v) for i, v in enumerate(chinese_digis)]
    digits[0].alt_s, digits[0].alt_t = ZERO_ALT, ZERO_ALT
    digits[1].alt_s, digits[1].alt_t = ONE_ALT, ONE_ALT
    digits[2].alt_s, digits[2].alt_t = TWO_ALTS[0], TWO_ALTS[1]

    # symbols
    positive_cn = CM(POSITIVE[0], POSITIVE[1], "+", lambda x: x)
    negative_cn = CM(NEGATIVE[0], NEGATIVE[1], "-", lambda x: -x)
    point_cn = CM(POINT[0], POINT[1], ".", lambda x, y: float(str(x) + "." + str(y)))
    # sil_cn = CM(SIL[0], SIL[1], '-', lambda x, y: float(str(x) + '-' + str(y)))
    system = NumberSystem()
    system.units = smaller_units + larger_units
    system.digits = digits
    system.math = MathSymbol(positive_cn, negative_cn, point_cn)
    # system.symbols = OtherSymbol(sil_cn)
    return system


def chn2num(chinese_string, numbering_type=NUMBERING_TYPES[1]):
    def get_symbol(char, system):
        for u in system.units:
            if char in [u.traditional, u.simplified, u.big_s, u.big_t]:
                return u
        for d in system.digits:
            if char in [d.traditional, d.simplified, d.big_s, d.big_t, d.alt_s, d.alt_t]:
                return d
        for m in system.math:
            if char in [m.traditional, m.simplified]:
                return m

    def string2symbols(chinese_string, system):
        int_string, dec_string = chinese_string, ""
        for p in [system.math.point.simplified, system.math.point.traditional]:
            if p in chinese_string:
                int_string, dec_string = chinese_string.split(p)
                break
        return [get_symbol(c, system) for c in int_string], [get_symbol(c, system) for c in dec_string]

    def correct_symbols(integer_symbols, system):
        """
        一百八 to 一百八十
        一亿一千三百万 to 一亿 一千万 三百万
        """

        if integer_symbols and isinstance(integer_symbols[0], CNU):
            if integer_symbols[0].power == 1:
                integer_symbols = [system.digits[1]] + integer_symbols

        if len(integer_symbols) > 1:
            if isinstance(integer_symbols[-1], CND) and isinstance(integer_symbols[-2], CNU):
                integer_symbols.append(CNU(integer_symbols[-2].power - 1, None, None, None, None))

        result = []
        unit_count = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                result.append(s)
                unit_count = 0
            elif isinstance(s, CNU):
                current_unit = CNU(s.power, None, None, None, None)
                unit_count += 1

            if unit_count == 1:
                result.append(current_unit)
            elif unit_count > 1:
                for i in range(len(result)):
                    if isinstance(result[-i - 1], CNU) and result[-i - 1].power < current_unit.power:
                        result[-i - 1] = CNU(result[-i - 1].power + current_unit.power, None, None, None, None)
        return result

    def compute_value(integer_symbols):
        """
        Compute the value.
        When current unit is larger than previous unit, current unit * all previous units will be used as all previous units.
        e.g. '两千万' = 2000 * 10000 not 2000 + 10000
        """
        value = [0]
        last_power = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                value[-1] = s.value
            elif isinstance(s, CNU):
                value[-1] *= pow(10, s.power)
                if s.power > last_power:
                    value[:-1] = list(map(lambda v: v * pow(10, s.power), value[:-1]))
                    last_power = s.power
                value.append(0)
        return sum(value)

    system = create_system(numbering_type)
    int_part, dec_part = string2symbols(chinese_string, system)
    int_part = correct_symbols(int_part, system)
    int_str = str(compute_value(int_part))
    dec_str = "".join([str(d.value) for d in dec_part])
    if dec_part:
        return "{0}.{1}".format(int_str, dec_str)
    else:
        return int_str


def num2chn(
    number_string,
    numbering_type=NUMBERING_TYPES[1],
    big=False,
    traditional=False,
    alt_zero=False,
    alt_one=False,
    alt_two=True,
    use_zeros=True,
    use_units=True,
):
    def get_value(value_string, use_zeros=True):
        striped_string = value_string.lstrip("0")

        # record nothing if all zeros
        if not striped_string:
            return []

        # record one digits
        elif len(striped_string) == 1:
            if use_zeros and len(value_string) != len(striped_string):
                return [system.digits[0], system.digits[int(striped_string)]]
            else:
                return [system.digits[int(striped_string)]]

        # recursively record multiple digits
        else:
            result_unit = next(u for u in reversed(system.units) if u.power < len(striped_string))
            result_string = value_string[: -result_unit.power]
            return get_value(result_string) + [result_unit] + get_value(striped_string[-result_unit.power :])

    system = create_system(numbering_type)

    int_dec = number_string.split(".")
    if len(int_dec) == 1:
        int_string = int_dec[0]
        dec_string = ""
    elif len(int_dec) == 2:
        int_string = int_dec[0]
        dec_string = int_dec[1]
    else:
        raise ValueError("invalid input num string with more than one dot: {}".format(number_string))

    if use_units and len(int_string) > 1:
        result_symbols = get_value(int_string)
    else:
        result_symbols = [system.digits[int(c)] for c in int_string]
    dec_symbols = [system.digits[int(c)] for c in dec_string]
    if dec_string:
        result_symbols += [system.math.point] + dec_symbols

    if alt_two:
        liang = CND(2, system.digits[2].alt_s, system.digits[2].alt_t, system.digits[2].big_s, system.digits[2].big_t)
        for i, v in enumerate(result_symbols):
            if isinstance(v, CND) and v.value == 2:
                next_symbol = result_symbols[i + 1] if i < len(result_symbols) - 1 else None
                previous_symbol = result_symbols[i - 1] if i > 0 else None
                if isinstance(next_symbol, CNU) and isinstance(previous_symbol, (CNU, type(None))):
                    if next_symbol.power != 1 and ((previous_symbol is None) or (previous_symbol.power != 1)):
                        result_symbols[i] = liang

    # if big is True, '两' will not be used and `alt_two` has no impact on output
    if big:
        attr_name = "big_"
        if traditional:
            attr_name += "t"
        else:
            attr_name += "s"
    else:
        if traditional:
            attr_name = "traditional"
        else:
            attr_name = "simplified"

    result = "".join([getattr(s, attr_name) for s in result_symbols])

    # if not use_zeros:
    #     result = result.strip(getattr(system.digits[0], attr_name))

    if alt_zero:
        result = result.replace(getattr(system.digits[0], attr_name), system.digits[0].alt_s)

    if alt_one:
        result = result.replace(getattr(system.digits[1], attr_name), system.digits[1].alt_s)

    for i, p in enumerate(POINT):
        if result.startswith(p):
            return CHINESE_DIGIS[0] + result

    # ^10, 11, .., 19
    if (
        len(result) >= 2
        and result[1] in [SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED[0], SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL[0]]
        and result[0] in [CHINESE_DIGIS[1], BIG_CHINESE_DIGIS_SIMPLIFIED[1], BIG_CHINESE_DIGIS_TRADITIONAL[1]]
    ):
        result = result[1:]

    return result


# ================================================================================ #
#                          different types of rewriters
# ================================================================================ #
class Cardinal:
    """
    CARDINAL类
    """

    def __init__(self, cardinal=None, chntext=None):
        self.cardinal = cardinal
        self.chntext = chntext

    def chntext2cardinal(self):
        return chn2num(self.chntext)

    def cardinal2chntext(self):
        return num2chn(self.cardinal)


class Digit:
    """
    DIGIT类
    """

    def __init__(self, digit=None, chntext=None):
        self.digit = digit
        self.chntext = chntext

    # def chntext2digit(self):
    #     return chn2num(self.chntext)

    def digit2chntext(self):
        return num2chn(self.digit, alt_two=False, use_units=False)


class TelePhone:
    """
    TELEPHONE类
    """

    def __init__(self, telephone=None, raw_chntext=None, chntext=None):
        self.telephone = telephone
        self.raw_chntext = raw_chntext
        self.chntext = chntext

    # def chntext2telephone(self):
    #     sil_parts = self.raw_chntext.split('<SIL>')
    #     self.telephone = '-'.join([
    #         str(chn2num(p)) for p in sil_parts
    #     ])
    #     return self.telephone

    def telephone2chntext(self, fixed=False):
        if fixed:
            sil_parts = self.telephone.split("-")
            self.raw_chntext = "<SIL>".join([num2chn(part, alt_two=False, use_units=False) for part in sil_parts])
            self.chntext = self.raw_chntext.replace("<SIL>", "")
        else:
            sp_parts = self.telephone.strip("+").split()
            self.raw_chntext = "<SP>".join([num2chn(part, alt_two=False, use_units=False) for part in sp_parts])
            self.chntext = self.raw_chntext.replace("<SP>", "")
        return self.chntext


class Fraction:
    """
    FRACTION类
    """

    def __init__(self, fraction=None, chntext=None):
        self.fraction = fraction
        self.chntext = chntext

    def chntext2fraction(self):
        denominator, numerator = self.chntext.split("分之")
        return chn2num(numerator) + "/" + chn2num(denominator)

    def fraction2chntext(self):
        numerator, denominator = self.fraction.split("/")
        return num2chn(denominator) + "分之" + num2chn(numerator)


class Date:
    """
    DATE类
    """

    def __init__(self, date=None, chntext=None):
        self.date = date
        self.chntext = chntext

    # def chntext2date(self):
    #     chntext = self.chntext
    #     try:
    #         year, other = chntext.strip().split('年', maxsplit=1)
    #         year = Digit(chntext=year).digit2chntext() + '年'
    #     except ValueError:
    #         other = chntext
    #         year = ''
    #     if other:
    #         try:
    #             month, day = other.strip().split('月', maxsplit=1)
    #             month = Cardinal(chntext=month).chntext2cardinal() + '月'
    #         except ValueError:
    #             day = chntext
    #             month = ''
    #         if day:
    #             day = Cardinal(chntext=day[:-1]).chntext2cardinal() + day[-1]
    #     else:
    #         month = ''
    #         day = ''
    #     date = year + month + day
    #     self.date = date
    #     return self.date

    def date2chntext(self):
        date = self.date
        try:
            year, other = date.strip().split("年", 1)
            year = Digit(digit=year).digit2chntext() + "年"
        except ValueError:
            other = date
            year = ""
        if other:
            try:
                month, day = other.strip().split("月", 1)
                month = Cardinal(cardinal=month).cardinal2chntext() + "月"
            except ValueError:
                day = date
                month = ""
            if day:
                day = Cardinal(cardinal=day[:-1]).cardinal2chntext() + day[-1]
        else:
            month = ""
            day = ""
        chntext = year + month + day
        self.chntext = chntext
        return self.chntext


class Money:
    """
    MONEY类
    """

    def __init__(self, money=None, chntext=None):
        self.money = money
        self.chntext = chntext

    # def chntext2money(self):
    #     return self.money

    def money2chntext(self):
        money = self.money
        pattern = re.compile(r"(\d+(\.\d+)?)")
        matchers = pattern.findall(money)
        if matchers:
            for matcher in matchers:
                money = money.replace(matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext())
        self.chntext = money
        return self.chntext


class Percentage:
    """
    PERCENTAGE类
    """

    def __init__(self, percentage=None, chntext=None):
        self.percentage = percentage
        self.chntext = chntext

    def chntext2percentage(self):
        return chn2num(self.chntext.strip().strip("百分之")) + "%"

    def percentage2chntext(self):
        return "百分之" + num2chn(self.percentage.strip().strip("%"))


def normalize_nsw(raw_text):
    text = "^" + raw_text + "$"

    # 规范化日期
    pattern = re.compile(r"\D+((([089]\d|(19|20)\d{2})年)?(\d{1,2}月(\d{1,2}[日号])?)?)")
    matchers = pattern.findall(text)
    if matchers:
        # print('date')
        for matcher in matchers:
            text = text.replace(matcher[0], Date(date=matcher[0]).date2chntext(), 1)

    # 规范化金钱
    pattern = re.compile(r"\D+((\d+(\.\d+)?)[多余几]?" + CURRENCY_UNITS + r"(\d" + CURRENCY_UNITS + r"?)?)")
    matchers = pattern.findall(text)
    if matchers:
        # print('money')
        for matcher in matchers:
            text = text.replace(matcher[0], Money(money=matcher[0]).money2chntext(), 1)

    # 规范化固话/手机号码
    # 手机
    # http://www.jihaoba.com/news/show/13680
    # 移动：139、138、137、136、135、134、159、158、157、150、151、152、188、187、182、183、184、178、198
    # 联通：130、131、132、156、155、186、185、176
    # 电信：133、153、189、180、181、177
    pattern = re.compile(r"\D((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})\D")
    matchers = pattern.findall(text)
    if matchers:
        # print('telephone')
        for matcher in matchers:
            text = text.replace(matcher[0], TelePhone(telephone=matcher[0]).telephone2chntext(), 1)
    # 固话
    pattern = re.compile(r"\D((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})\D")
    matchers = pattern.findall(text)
    if matchers:
        # print('fixed telephone')
        for matcher in matchers:
            text = text.replace(matcher[0], TelePhone(telephone=matcher[0]).telephone2chntext(fixed=True), 1)

    # 规范化分数
    pattern = re.compile(r"(\d+/\d+)")
    matchers = pattern.findall(text)
    if matchers:
        # print('fraction')
        for matcher in matchers:
            text = text.replace(matcher, Fraction(fraction=matcher).fraction2chntext(), 1)

    # 规范化百分数
    text = text.replace("％", "%")
    pattern = re.compile(r"(\d+(\.\d+)?%)")
    matchers = pattern.findall(text)
    if matchers:
        # print('percentage')
        for matcher in matchers:
            text = text.replace(matcher[0], Percentage(percentage=matcher[0]).percentage2chntext(), 1)

    # 规范化纯数+量词
    pattern = re.compile(r"(\d+(\.\d+)?)[多余几]?" + COM_QUANTIFIERS)
    matchers = pattern.findall(text)
    if matchers:
        # print('cardinal+quantifier')
        for matcher in matchers:
            text = text.replace(matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext(), 1)

    # 规范化数字编号
    pattern = re.compile(r"(\d{4,32})")
    matchers = pattern.findall(text)
    if matchers:
        # print('digit')
        for matcher in matchers:
            text = text.replace(matcher, Digit(digit=matcher).digit2chntext(), 1)

    # 规范化纯数
    pattern = re.compile(r"(\d+(\.\d+)?)")
    matchers = pattern.findall(text)
    if matchers:
        # print('cardinal')
        for matcher in matchers:
            text = text.replace(matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext(), 1)

    # restore P2P, O2O, B2C, B2B etc
    pattern = re.compile(r"(([a-zA-Z]+)二([a-zA-Z]+))")
    matchers = pattern.findall(text)
    if matchers:
        # print('particular')
        for matcher in matchers:
            text = text.replace(matcher[0], matcher[1] + "2" + matcher[2], 1)

    return text.lstrip("^").rstrip("$")


def remove_erhua(text):
    """
    去除儿化音词中的儿:
    他女儿在那边儿 -> 他女儿在那边
    """

    new_str = ""
    while re.search("儿", text):
        a = re.search("儿", text).span()
        remove_er_flag = 0

        if ER_WHITELIST_PATTERN.search(text):
            b = ER_WHITELIST_PATTERN.search(text).span()
            if b[0] <= a[0]:
                remove_er_flag = 1

        if remove_er_flag == 0:
            new_str = new_str + text[0 : a[0]]
            text = text[a[1] :]
        else:
            new_str = new_str + text[0 : b[1]]
            text = text[b[1] :]

    text = new_str + text
    return text


def remove_space(text):
    tokens = text.split()
    new = []
    for k, t in enumerate(tokens):
        if k != 0:
            if IN_EN_CHARS.get(tokens[k - 1][-1]) and IN_EN_CHARS.get(t[0]):
                new.append(" ")
        new.append(t)
    return "".join(new)


class TextNorm:
    def __init__(
        self,
        to_banjiao: bool = False,
        to_upper: bool = False,
        to_lower: bool = False,
        remove_fillers: bool = False,
        remove_erhua: bool = False,
        check_chars: bool = False,
        remove_space: bool = False,
        cc_mode: str = "",
    ):
        self.to_banjiao = to_banjiao
        self.to_upper = to_upper
        self.to_lower = to_lower
        self.remove_fillers = remove_fillers
        self.remove_erhua = remove_erhua
        self.check_chars = check_chars
        self.remove_space = remove_space

        self.cc = None
        if cc_mode:
            from opencc import OpenCC  # Open Chinese Convert: pip install opencc

            self.cc = OpenCC(cc_mode)

    def __call__(self, text):
        if self.cc:
            text = self.cc.convert(text)

        if self.to_banjiao:
            text = text.translate(QJ2BJ_TRANSFORM)

        if self.to_upper:
            text = text.upper()

        if self.to_lower:
            text = text.lower()

        if self.remove_fillers:
            for c in FILLER_CHARS:
                text = text.replace(c, "")

        if self.remove_erhua:
            text = remove_erhua(text)

        text = normalize_nsw(text)

        text = text.translate(PUNCS_TRANSFORM)

        if self.check_chars:
            for c in text:
                if not IN_VALID_CHARS.get(c):
                    print(f"WARNING: illegal char {c} in: {text}", file=sys.stderr)
                    return ""

        if self.remove_space:
            text = remove_space(text)

        return text


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # normalizer options
    p.add_argument("--to_banjiao", action="store_true", help="convert quanjiao chars to banjiao")
    p.add_argument("--to_upper", action="store_true", help="convert to upper case")
    p.add_argument("--to_lower", action="store_true", help="convert to lower case")
    p.add_argument("--remove_fillers", action="store_true", help='remove filler chars such as "呃, 啊"')
    p.add_argument("--remove_erhua", action="store_true", help='remove erhua chars such as "他女儿在那边儿 -> 他女儿在那边"')
    p.add_argument("--check_chars", action="store_true", help="skip sentences containing illegal chars")
    p.add_argument("--remove_space", action="store_true", help="remove whitespace")
    p.add_argument(
        "--cc_mode", choices=["", "t2s", "s2t"], default="", help="convert between traditional to simplified"
    )

    # I/O options
    p.add_argument("--log_interval", type=int, default=10000, help="log interval in number of processed lines")
    p.add_argument("--has_key", action="store_true", help="will be deprecated, set --format ark instead")
    p.add_argument("--format", type=str, choices=["txt", "ark", "tsv"], default="txt", help="input format")
    p.add_argument("ifile", help="input filename, assume utf-8 encoding")
    p.add_argument("ofile", help="output filename")

    args = p.parse_args()

    if args.has_key:
        args.format = "ark"

    normalizer = TextNorm(
        to_banjiao=args.to_banjiao,
        to_upper=args.to_upper,
        to_lower=args.to_lower,
        remove_fillers=args.remove_fillers,
        remove_erhua=args.remove_erhua,
        check_chars=args.check_chars,
        remove_space=args.remove_space,
        cc_mode=args.cc_mode,
    )

    normalizer = TextNorm(
        to_banjiao=args.to_banjiao,
        to_upper=args.to_upper,
        to_lower=args.to_lower,
        remove_fillers=args.remove_fillers,
        remove_erhua=args.remove_erhua,
        check_chars=args.check_chars,
        remove_space=args.remove_space,
        cc_mode=args.cc_mode,
    )

    ndone = 0
    with open(args.ifile, "r", encoding="utf8") as istream, open(args.ofile, "w+", encoding="utf8") as ostream:
        if args.format == "tsv":
            reader = csv.DictReader(istream, delimiter="\t")
            assert "TEXT" in reader.fieldnames
            print("\t".join(reader.fieldnames), file=ostream)

            for item in reader:
                text = item["TEXT"]

                if text:
                    text = normalizer(text)

                if text:
                    item["TEXT"] = text
                    print("\t".join([item[f] for f in reader.fieldnames]), file=ostream)

                ndone += 1
                if ndone % args.log_interval == 0:
                    print(f"text norm: {ndone} lines done.", file=sys.stderr, flush=True)
        else:
            for l in istream:
                key, text = "", ""
                if args.format == "ark":  # KALDI archive, line format: "key text"
                    cols = l.strip().split(maxsplit=1)
                    key, text = cols[0], cols[1] if len(cols) == 2 else ""
                else:
                    text = l.strip()

                if text:
                    text = normalizer(text)

                if text:
                    if args.format == "ark":
                        print(key + "\t" + text, file=ostream)
                    else:
                        print(text, file=ostream)

                ndone += 1
                if ndone % args.log_interval == 0:
                    print(f"text norm: {ndone} lines done.", file=sys.stderr, flush=True)
    print(f"text norm: {ndone} lines done in total.", file=sys.stderr, flush=True)
