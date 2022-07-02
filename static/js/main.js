$(document).ready(function () {
  $("#predict-btn").hide();
  $("#loading").hide();

  function getFileUrl(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function (e) {
        $("#imageWidget").attr("src", e.target.result);
        $("#result").html("<div></div>");
      };
      reader.readAsDataURL(input.files[0]);
    }
  }
  $("#image-input").change(function () {
    getFileUrl(this);
    $("#imageWidget").show();
    $("#predict-btn").show();
  });

  $("#predict-btn").click(() => {
    $("#loading").show();
    let uploadedImage = new FormData($("#upload-form")[0]);

    $.ajax({
      type: "POST",
      url: "/predict",
      data: uploadedImage,
      contentType: false,
      cache: false,
      processData: false,
      async: true,
      success: function (data) {
        $("#loading").hide();
        btn = "<button class='btn btn-success'>Not Ictal</button>";
        if (data["status"] == 0) {
          btn = "<button class='btn btn-danger'>Ictal</button>";
        }

        $("#result").html(btn);
        console.log(data);
      },
    });
  });
});
