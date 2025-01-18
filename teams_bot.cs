using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Integration.AspNet.Core;
using Microsoft.Bot.Schema;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

public class TeamsBot : ActivityHandler
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly string _apiBaseUrl;

    public TeamsBot(IHttpClientFactory httpClientFactory, IConfiguration configuration)
    {
        _httpClientFactory = httpClientFactory;
        _apiBaseUrl = configuration["ApiBaseUrl"];
    }

    protected override async Task OnMessageActivityAsync(ITurnContext<IMessageActivity> turnContext, CancellationToken cancellationToken)
    {
        var userInput = turnContext.Activity.Text;

        if (userInput.StartsWith("upload"))
        {
            // Handle document upload
            var fileData = await DownloadFileAsync(turnContext.Activity.Attachments[0].ContentUrl);
            var extractionResult = await ExtractTextAsync(fileData);
            await turnContext.SendActivityAsync(MessageFactory.Text("Document uploaded and processed successfully."));
            await turnContext.SendActivityAsync(MessageFactory.Text($"Extracted text: {extractionResult["chunks"][0]}..."));
        }
        else if (userInput.StartsWith("question"))
        {
            // Handle question submission
            var question = userInput.Substring("question ".Length);
            var summaries = new Dictionary<string, string>(); // Fetch summaries from your storage
            var relevantDoc = await SelectRelevantAsync(question, summaries);
            var answer = await GetAnswerAsync(question, relevantDoc["most_relevant"]);
            await turnContext.SendActivityAsync(MessageFactory.Text($"Answer: {answer}"));
        }
        else
        {
            await turnContext.SendActivityAsync(MessageFactory.Text("Please upload a document or ask a question."));
        }
    }

    private async Task<byte[]> DownloadFileAsync(string fileUrl)
    {
        var client = _httpClientFactory.CreateClient();
        return await client.GetByteArrayAsync(fileUrl);
    }

    private async Task<Dictionary<string, object>> ExtractTextAsync(byte[] fileData)
    {
        var client = _httpClientFactory.CreateClient();
        using var content = new MultipartFormDataContent();
        content.Add(new ByteArrayContent(fileData), "file", "document.pdf");
        var response = await client.PostAsync($"{_apiBaseUrl}/extract_text/", content);
        response.EnsureSuccessStatusCode();
        var responseString = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<Dictionary<string, object>>(responseString);
    }

    private async Task<Dictionary<string, object>> SelectRelevantAsync(string question, Dictionary<string, string> summaries)
    {
        var client = _httpClientFactory.CreateClient();
        var content = new StringContent(JsonConvert.SerializeObject(new { question, summaries }), System.Text.Encoding.UTF8, "application/json");
        var response = await client.PostAsync($"{_apiBaseUrl}/select_relevant/", content);
        response.EnsureSuccessStatusCode();
        var responseString = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<Dictionary<string, object>>(responseString);
    }

    private async Task<string> GetAnswerAsync(string question, string documentText)
    {
        var client = _httpClientFactory.CreateClient();
        var content = new StringContent(JsonConvert.SerializeObject(new { question, documentText }), System.Text.Encoding.UTF8, "application/json");
        var response = await client.PostAsync($"{_apiBaseUrl}/get_answer/", content);
        response.EnsureSuccessStatusCode();
        var responseString = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<Dictionary<string, string>>(responseString)["answer"];
    }
}

public class AdapterWithErrorHandler : BotFrameworkHttpAdapter
{
    public AdapterWithErrorHandler(IConfiguration configuration, ILogger<BotFrameworkHttpAdapter> logger)
        : base(configuration, logger)
    {
        OnTurnError = async (turnContext, exception) =>
        {
            logger.LogError($"Exception caught : {exception}");
            await turnContext.SendActivityAsync(MessageFactory.Text("Sorry, it looks like something went wrong."));
        };
    }
}

public class Startup
{
    public IConfiguration Configuration { get; }

    public Startup(IConfiguration configuration)
    {
        Configuration = configuration;
    }

    public void ConfigureServices(IServiceCollection services)
    {
        services.AddHttpClient();
        services.AddControllers().AddNewtonsoftJson();
        services.AddSingleton<IBotFrameworkHttpAdapter, AdapterWithErrorHandler>();
        services.AddTransient<IBot, TeamsBot>();
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        if (env.IsDevelopment())
        {
            app.UseDeveloperExceptionPage();
        }

        app.UseDefaultFiles();
        app.UseStaticFiles();
        app.UseRouting();
        app.UseAuthorization();
        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllers();
        });
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        CreateHostBuilder(args).Build().Run();
    }

    public static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<Startup>();
            });
}
