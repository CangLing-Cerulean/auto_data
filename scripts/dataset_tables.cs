using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

// Lossless numeric-token packaging only. No learning, scoring, filtering or imputation.
public static class DatasetTables {
    static readonly CultureInfo CI = CultureInfo.InvariantCulture;
    static readonly UTF8Encoding UTF8 = new UTF8Encoding(false, true);
    public sealed class Row {
        public string Source;
        public int SourceRow, Sample;
        public string Time;
        public string[] Values;
        public string Key;
    }
    public static string HashFile(string path) {
        using (var s = File.OpenRead(path)) return Convert.ToHexString(SHA256.HashData(s)).ToLowerInvariant();
    }
    static double Number(string token) {
        if (!double.TryParse(token, NumberStyles.Float, CI, out double x) || !double.IsFinite(x))
            throw new InvalidDataException("Missing/nonfinite sensor value");
        return x;
    }
    public static List<Row> Skab(string root, string[] sensors) {
        var rows = new List<Row>();
        var paths = Directory.GetFiles(root, "*.csv", SearchOption.AllDirectories);
        Array.Sort(paths, StringComparer.Ordinal);
        foreach (var path in paths) {
            string source = Path.GetRelativePath(root, path).Replace('\\','/');
            using (var reader = new StreamReader(path, UTF8)) {
                var header = reader.ReadLine().Split(';');
                if (!header.Take(9).SequenceEqual(new[]{"datetime"}.Concat(sensors)) ||
                    !(header.Length == 9 || (header.Length == 11 && header[9] == "anomaly" && header[10] == "changepoint")))
                    throw new InvalidDataException("Unexpected SKAB schema: " + source);
                int n = 0; DateTime? previous = null; string line;
                while ((line = reader.ReadLine()) != null) {
                    var fields = line.Split(';');
                    if (fields.Length != header.Length) throw new InvalidDataException("SKAB field count");
                    var time = DateTime.ParseExact(fields[0], "yyyy-MM-dd HH:mm:ss", CI);
                    if (previous.HasValue && time <= previous.Value) throw new InvalidDataException("SKAB chronology");
                    previous = time;
                    var values = fields.Skip(1).Take(8).ToArray();
                    foreach (var v in values) Number(v);
                    rows.Add(new Row {Source=source, SourceRow=n++, Sample=-1, Time=fields[0], Values=values, Key=fields[0]});
                }
            }
        }
        // Overlapping trials remain separate. Ordering uses only timestamps and provenance.
        rows.Sort((a,b) => {
            int c = StringComparer.Ordinal.Compare(a.Time,b.Time);
            if (c != 0) return c;
            c = StringComparer.Ordinal.Compare(a.Source,b.Source);
            return c != 0 ? c : a.SourceRow.CompareTo(b.SourceRow);
        });
        return rows;
    }
    public static List<Row> Hydraulic(string root, string[] sensors) {
        var matrices = sensors.Select(s => File.ReadAllLines(Path.Combine(root,s+".txt"), UTF8)
            .Select(line => line.Split((char[])null,StringSplitOptions.RemoveEmptyEntries)).ToArray()).ToArray();
        int cycles = matrices[0].Length;
        if (cycles != 2205 || matrices.Any(m => m.Length != cycles || m.Any(r => r.Length != 60)))
            throw new InvalidDataException("Hydraulic files must align: 2205 cycles x 60 samples");
        var rows = new List<Row>(cycles*60);
        for (int cycle=0; cycle<cycles; cycle++) for (int sample=0; sample<60; sample++) {
            var values = matrices.Select(m => m[cycle][sample]).ToArray();
            foreach (var v in values) Number(v);
            rows.Add(new Row {Source="", SourceRow=cycle, Sample=sample, Values=values,
                Key=cycle.ToString("D6",CI)+":"+sample.ToString("D2",CI)});
        }
        return rows;
    }
    static string Line(Row r, bool skab) {
        return (skab ? r.Source+","+r.SourceRow.ToString(CI)+","+r.Time :
            r.SourceRow.ToString(CI)+","+r.Sample.ToString(CI)) + "," + string.Join(",",r.Values);
    }
    public static Dictionary<string,object> Package(List<Row> rows, string[] sensors, string output, bool skab, bool verify) {
        int n=rows.Count, train=n*7/10, validation=n*2/10;
        int[] starts={0,train,train+validation}, ends={train,train+validation,n};
        string[] names={"train","validation","test"};
        var splits=new List<object>();
        string header=(skab?"source_file,source_row,datetime,":"cycle_id,sample_offset_seconds,")+string.Join(",",sensors);
        var statistics = sensors.ToDictionary(s=>s,s=>new double[]{double.PositiveInfinity,double.NegativeInfinity,0});
        using (var combined=IncrementalHash.CreateHash(HashAlgorithmName.SHA256)) {
            combined.AppendData(UTF8.GetBytes(header+"\n"));
            for (int part=0; part<3; part++) {
                if (starts[part]>=ends[part]) throw new InvalidDataException("Empty partition");
                if (part>0 && StringComparer.Ordinal.Compare(rows[starts[part]-1].Key, rows[starts[part]].Key)>=0)
                    throw new InvalidDataException("Equal/reversed split time boundary");
                string file=names[part]+".csv", path=Path.Combine(output,file);
                var series=new Dictionary<string,List<Row>>(StringComparer.Ordinal);
                var intervals=new SortedDictionary<string,int>(StringComparer.Ordinal);
                var previous=new Dictionary<string,Row>(StringComparer.Ordinal);
                StreamWriter writer = verify ? null : new StreamWriter(path,false,UTF8);
                if (writer != null) writer.NewLine="\n";
                string hash;
                using (var digest=IncrementalHash.CreateHash(HashAlgorithmName.SHA256)) {
                    byte[] h=UTF8.GetBytes(header+"\n"); digest.AppendData(h);
                    try {
                        if (writer!=null) writer.WriteLine(header);
                        for (int i=starts[part]; i<ends[part]; i++) {
                            Row row=rows[i]; byte[] data=UTF8.GetBytes(Line(row,skab)+"\n");
                            digest.AppendData(data); combined.AppendData(data);
                            if (writer!=null) writer.WriteLine(Line(row,skab));
                            string group=skab?row.Source:row.SourceRow.ToString(CI);
                            if (!series.ContainsKey(group)) series[group]=new List<Row>();
                            series[group].Add(row);
                            if (previous.TryGetValue(group,out Row prior)) {
                                int delta=skab?(int)(DateTime.ParseExact(row.Time,"yyyy-MM-dd HH:mm:ss",CI)-
                                    DateTime.ParseExact(prior.Time,"yyyy-MM-dd HH:mm:ss",CI)).TotalSeconds:row.Sample-prior.Sample;
                                if (delta<=0) throw new InvalidDataException("Nonincreasing within-series time");
                                string key=delta.ToString(CI); if (!intervals.ContainsKey(key)) intervals[key]=0; intervals[key]++;
                            }
                            previous[group]=row;
                            if (part==0) for (int j=0;j<sensors.Length;j++) {
                                double x=Number(row.Values[j]); var stat=statistics[sensors[j]];
                                stat[0]=Math.Min(stat[0],x);stat[1]=Math.Max(stat[1],x);stat[2]+=x;
                            }
                        }
                    } finally { if(writer!=null) writer.Dispose(); }
                    hash=Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant();
                }
                if (HashFile(path)!=hash) throw new InvalidDataException("Source-token reconstruction differs: "+file);
                var coverage=series.OrderBy(k=>k.Key,StringComparer.Ordinal).Select(k=>(object)new Dictionary<string,object>{
                    {"series_id",k.Key},{"rows",k.Value.Count},
                    {"source_row_start_inclusive",k.Value[0].SourceRow},{"source_row_end_exclusive",k.Value.Last().SourceRow+1},
                    {"first_time",skab?k.Value[0].Time:(object)k.Value[0].Sample},
                    {"last_time",skab?k.Value.Last().Time:(object)k.Value.Last().Sample}
                }).ToList();
                splits.Add(new Dictionary<string,object>{{"name",names[part]},{"file",file},{"rows",ends[part]-starts[part]},
                    {"row_start_inclusive",starts[part]},{"row_end_exclusive",ends[part]},{"sha256",hash},
                    {"first_order_key",rows[starts[part]].Key},{"last_order_key",rows[ends[part]-1].Key},
                    {"within_series_interval_seconds_counts",intervals},{"series_coverage",coverage}});
            }
            var profile=statistics.ToDictionary(k=>k.Key,k=>(object)new Dictionary<string,object>{
                {"min",k.Value[0]},{"max",k.Value[1]},{"missing",0},{"mean",k.Value[2]/train}});
            return new Dictionary<string,object>{{"rows",n},{"columns_raw",header.Split(',')},{"splits",splits},
                {"table_sha256",Convert.ToHexString(combined.GetHashAndReset()).ToLowerInvariant()},{"train_profile",profile}};
        }
    }
    public static string ReconstructMetro(string root) {
        using (var digest=IncrementalHash.CreateHash(HashAlgorithmName.SHA256)) {
            byte[] buffer=new byte[1024*1024];
            foreach (var name in new[]{"train","validation","test"}) {
                using (var stream=File.OpenRead(Path.Combine(root,name+".csv"))) {
                    if(name!="train") { int c; do {c=stream.ReadByte();} while(c!=-1 && c!=10); }
                    int length; while((length=stream.Read(buffer,0,buffer.Length))>0) digest.AppendData(buffer,0,length);
                }
            }
            return Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant();
        }
    }
}
