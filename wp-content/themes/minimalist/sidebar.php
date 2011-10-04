<div id="leftcol" >
<div id="menu">

<h2 class="menuheader">Archives</h2>
<div class="menucontent">
<ul>
<?php wp_get_archives('type=monthly'); ?>
</ul>
</div>


<h2 class="menuheader">Links</h2>
<div class="menucontent">
<ul>
<?php wp_list_bookmarks('title_li=&categorize=0'); ?>
</ul>
</div>

<?php if ( !function_exists('dynamic_sidebar')         || !dynamic_sidebar() ); ?>

<p>
<a href="<?php bloginfo('atom_url'); ?>" title="<?php _e('Syndicate this site using Atom'); ?>">
atom
</a>|
<a href="<?php bloginfo('rss2_url'); ?>" title="<?php _e('Syndicate this site using RSS'); ?>">
rss
</a>|
<a href="http://twitter.com/avisheksengupta">twitter</a> |
<a href="http://github.com/asengupta">github</a>
</p>

</div>
</div>