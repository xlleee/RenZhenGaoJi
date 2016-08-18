%let codeloc = E:\work\dailytrack\database\;

%include "E:\work\dailytrack\code\common.sas";run;
%include "E:\work\dailytrack\code\macro_utility.sas";run;
%include "E:\work\dailytrack\code\macro_quant.sas";run;

%let dir = &codeloc;

/*开始结束时间*/
/*%let begDate = 2011-01-01;*/
/*%let endDate = 2012-07-17;*/
%let begDateTime = "01Dec2010:00:00:00"dt;
%let endDateTime = "31Dec2016:00:00:00"dt;

*读入今天的日期文件;
/*proc import datafile="&codeloc\date.txt" out=date REPLACE;run;*/
proc sql;
/*MF_FundArchives包含了基金的投资类型*/
/*TYPE 1-契约型封闭  2-开放式  3-LOF 4-ETF 5-FOF 6-创新型封闭式 7-开放式（带固定封闭期) 8-ETF联接 9-半开放式*/
/*InvestmentType 投资类型（jydb.CT_SystemConst LB=1094） 1.积极成长型 2.稳健成长性 3.中小企业成长型 4，平衡型 5.资产重组型 6.科技型 7.指数型 8.优化指数型 */
/*9.价值型 10.债券型 11.收益型 15.现金型 20.内需增长型 99.综合型 21.生命周期型*/
/*InvestStyle 投资风格 jydb.CT_SystemConst LB=1093*/
/*FundTypeCode 晨星分类 jydb.CT_SystemConst LB=1273; 1103-混合型 1105-债券型  1107-保本型  1109-货币型 1110-QDII 1199-其他型 1101-股票型*/
create table fundlist as
	select innercode,Type,InvestmentType,InvestStyle,FundTypeCode from jydb.MF_FundArchives
	where type in (2,3) and FundTypeCode ^= 1109;
create table fundmanager as
	select InnerCode,Name,EducationLevel,PracticeDate,AccessionDate,DimissionDate from jydb.MF_FundManager
	where PostName=1 and (DimissionDate =. or DimissionDate >=&begDateTime )and AccessionDate<=&endDateTime;

create table fundNAV as
	select InnerCode,EndDate,UnitNV
	from jydb.MF_NetValue where EndDate between &begDateTime and &endDateTime
	and innercode in (select innercode from fundlist);
/*create table fundDividend as*/
/*	select InnerCode,ExRightDate,DividendRatioBeforeTax*/
/*	from jydb.MF_Dividend where ExRightDate between &begDateTime and &endDateTime*/
/*	and innercode in (select innercode from fundlist);*/
/*create table fundShare as*/
/*	select distinct innercode,EndDate,EndShares from jydb.MF_SharesChange */
/*	where EndShares>0 and innercode in (select innercode from fundlist)*/
/*	group by innercode,EndDate having EndShares=min(EndShares)*/
/*	order by innercode,EndDate;*/
create table fundNAV as
	select c.SecuCode,c.Secuabbr,a.*,e.InvestAdvisorAbbrName,b.GrowthRateFactor,d.Type,d.InvestmentType,d.InvestStyle,d.FundTypeCode
	from fundNAV a
	left join jydb.secumain c on a.innercode=c.innercode
	left join fundlist d on a.innercode=d.innercode
	left join jydb.MF_AdjustingFactor b on a.innercode=b.innercode and a.EndDate=b.ExDiviDate
	left join (select innercode,aa.InvestAdvisorCode,InvestAdvisorAbbrName from jydb.MF_FundArchives aa left join jydb.MF_InvestAdvisorOutline bb
		on aa.InvestAdvisorCode=bb.InvestAdvisorCode) e on a.innercode=e.innercode
	order by innercode,EndDate;
/*create table fundNAV as*/
/*	select c.SecuCode,c.Secuabbr,a.*,b.DividendRatioBeforeTax,d.Type,d.InvestmentType,d.InvestStyle,d.FundTypeCode,e.EndShares,e.EndDate as SharesDate*/
/*	from fundNAV a left join fundDividend b on a.innercode=b.innercode and a.EndDate=b.ExRightDate*/
/*		left join jydb.secumain c on a.innercode=c.innercode */
/*		left join fundlist d on a.innercode=d.innercode */
/*		left join fundShare e on a.innercode=e.innercode and a.EndDate=e.EndDate */
/*	order by innercode,EndDate;*/
quit;
data fst;set fundNAV;if first.innercode;by innercode;run;
proc sql;create table fst as
select a.innercode,a.enddate,b.GrowthRateFactor
from fst a left join jydb.MF_AdjustingFactor b on a.innercode=b.InnerCode and a.enddate>=b.ExDiviDate
group by a.innercode having ExDiviDate=max(ExDiviDate);quit;
data fst;set fst;if GrowthRateFactor=. then GrowthRateFactor=1;run;
data fundNAV;update fundNAV fst;by innercode enddate;run;

data fundNAV;set fundNAV;retain t;
if GrowthRateFactor~=. then t=GrowthRateFactor;
else GrowthRateFactor=t;
drop t;
run;

data fundmanager;
	set fundmanager;
	format ManagerID $20.;
	length ManagerID $ 20;
	if PracticeDate=. then
		ManagerID=cats(Name,"00000000");
	else
		ManagerID=cats(substr(Name,1,10),put(datepart(PracticeDate),yymmddn8.));
run;
/*EndShares 还没处理好*/
data fundNAV(rename=(secuabbrtem=SecuAbbr) );
	set fundNAV;
	format secuabbrtem $20.;
	length secuabbrtem $ 20;
	secuabbrtem=SecuAbbr;
	UnitNVAdj = GrowthRateFactor*UnitNV;
	tt = lag(UnitNVAdj);
	if first.innercode then
		lastNAVAdj = .;
	else
		lastNAVAdj =tt;
	by innercode;
	dailyreturn = UnitNVAdj/lastNAVAdj-1;
	retain tt;
	drop tt SecuAbbr;
run;
data fundNAV;
	retain InnerCode SecuCode SecuAbbr;
	set fundNAV;
run;
proc sql;
create table FundAndManagerData as
	select ManagerID,Name,EducationLevel,PracticeDate,AccessionDate,DimissionDate,a.*
	from fundNAV a join FundManager b on a.innercode=b.innercode
	order by innercode,EndDate;
quit;
data FundAndManagerData;
	set FundAndManagerData;
	format ID $40.;
	length ID $ 40;
	if EndDate>=AccessionDate and (EndDate<DimissionDate or DimissionDate=.);
	ID=cats(ManagerID,'D',put(datepart(EndDate),yymmddn8.),'F',innercode);
run;
proc sql;
create table FundsofManager as
	select distinct ManagerID,EndDate,count(*) as FundsofManager
	from FundAndManagerData group by ManagerID,EndDate;
create table ManagersofFund as
	select distinct innercode,EndDate,count(*) as ManagersofFund
	from FundAndManagerData group by innercode,EndDate;
create table FundAndManagerData as
	select a.*,FundsofManager,ManagersofFund
	from FundAndManagerData a
		left join FundsofManager b on a.ManagerID=b.ManagerID and a.EndDate=b.EndDate
		left join ManagersofFund c on a.innercode=c.innercode and a.EndDate=c.EndDate
	order by innercode,EndDate;
quit;
data FundAndManagerData;
retain ID;
set FundAndManagerData;
run;
proc sql;
/*导入数据库*/
INSERT INTO jrgcb.FundAndManagerData
	select * from FundAndManagerData as BB
	where BB.id not in (select distinct id from jrgcb.FundAndManagerData where EndDate >=&begDateTime);
quit;


data tem;
set FundAndManagerData;
if managerid='陈逊20010101' and enddate in ('29MAY2015:00:00:00'dt ,'28MAY2015:00:00:00'dt);
run;
proc sql;
create table tem as
select * from jydb.CT_SystemConst
where LB=1093;
create table tem2 as
select * from jydb.secumain
where innercode=6790;
create table tem3 as
select * from jydb.MF_FundArchives
where innercode in (6790,19153);
create table tem4 as
	select InnerCode,Name,EducationLevel,PracticeDate,AccessionDate,DimissionDate from jydb.MF_FundManager
	where innercode=3025;
select distinct count(*) from FundAndManagerData where enddate<="31Dec2010:00:00:00"dt;
